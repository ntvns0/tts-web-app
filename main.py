from __future__ import annotations

import asyncio
import json
import queue
import re
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import edge_tts
import numpy as np
import soundfile as sf
from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from piper import PiperVoice
from piper.download_voices import download_voice
from pypdf import PdfReader
from rapidocr_onnxruntime import RapidOCR


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
JOBS_DIR = DATA_DIR / "jobs"
MODELS_DIR = BASE_DIR / "models"
MAX_FILE_SIZE = 5 * 1024 * 1024
CHUNK_LIMIT = 350
CHUNK_GAP_SECONDS = 0.18

VOICE_CATALOG: dict[str, dict[str, str]] = {
    "edge:en-US-EmmaMultilingualNeural": {
        "provider": "edge",
        "voice_name": "en-US-EmmaMultilingualNeural",
        "label": "Emma Multilingual Neural",
        "description": "Most natural, expressive female voice",
        "format": "mp3",
    },
    "edge:en-US-AndrewMultilingualNeural": {
        "provider": "edge",
        "voice_name": "en-US-AndrewMultilingualNeural",
        "label": "Andrew Multilingual Neural",
        "description": "Most natural, expressive male voice",
        "format": "mp3",
    },
    "edge:en-US-AvaMultilingualNeural": {
        "provider": "edge",
        "voice_name": "en-US-AvaMultilingualNeural",
        "label": "Ava Multilingual Neural",
        "description": "Natural conversational female voice",
        "format": "mp3",
    },
    "edge:en-US-BrianMultilingualNeural": {
        "provider": "edge",
        "voice_name": "en-US-BrianMultilingualNeural",
        "label": "Brian Multilingual Neural",
        "description": "Natural conversational male voice",
        "format": "mp3",
    },
    "piper:en_US-ryan-high": {
        "provider": "piper",
        "voice_name": "en_US-ryan-high",
        "label": "Ryan High",
        "description": "Best local offline male voice",
        "format": "wav",
    },
    "piper:en_US-lessac-medium": {
        "provider": "piper",
        "voice_name": "en_US-lessac-medium",
        "label": "Lessac Medium",
        "description": "Fast local fallback voice",
        "format": "wav",
    },
}
DEFAULT_VOICE = "edge:en-US-EmmaMultilingualNeural"

for directory in (STATIC_DIR, DATA_DIR, UPLOADS_DIR, AUDIO_DIR, TRANSCRIPTS_DIR, JOBS_DIR, MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class Job:
    id: str
    title: str
    source_type: str
    voice: str
    state: str
    text_length: int
    created_at: float
    updated_at: float
    original_filename: str | None = None
    provider: str = ""
    voice_label: str = ""
    progress: float = 0.0
    total_chunks: int = 0
    completed_chunks: int = 0
    selected: bool = False
    audio_url: str | None = None
    download_url: str | None = None
    transcript_url: str | None = None
    duration_seconds: float | None = None
    audio_format: str | None = None
    timing_mode: str | None = None
    error: str | None = None
    preview: str = ""
    transcript_path: str | None = None
    transcript_data_path: str | None = None
    audio_path: str | None = None
    library_name: str | None = None

    def payload(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("transcript_path", None)
        data.pop("transcript_data_path", None)
        data.pop("audio_path", None)
        return data


app = FastAPI(title="TTS Web App")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

jobs_lock = threading.Lock()
jobs: dict[str, Job] = {}
job_queue: queue.Queue[str] = queue.Queue()
voices_lock = threading.Lock()
piper_voices: dict[str, PiperVoice] = {}
worker_started = False
ocr_lock = threading.Lock()
ocr_engine: RapidOCR | None = None


def get_voice_config(voice_id: str) -> dict[str, str]:
    if voice_id not in VOICE_CATALOG:
        raise HTTPException(status_code=400, detail="Unknown voice selection.")
    return VOICE_CATALOG[voice_id]


def ensure_piper_voice(voice_name: str) -> Path:
    model_path = MODELS_DIR / f"{voice_name}.onnx"
    config_path = MODELS_DIR / f"{voice_name}.onnx.json"
    if model_path.exists() and config_path.exists():
        return model_path
    download_voice(voice_name, MODELS_DIR)
    return model_path


def get_piper_voice(voice_name: str) -> PiperVoice:
    with voices_lock:
        if voice_name not in piper_voices:
            model_path = ensure_piper_voice(voice_name)
            piper_voices[voice_name] = PiperVoice.load(model_path)
        return piper_voices[voice_name]


def get_ocr_engine() -> RapidOCR:
    global ocr_engine
    with ocr_lock:
        if ocr_engine is None:
            ocr_engine = RapidOCR()
        return ocr_engine


def clean_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized


def split_text(text: str, limit: int = CHUNK_LIMIT) -> list[str]:
    paragraphs = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    chunks: list[str] = []
    for paragraph in paragraphs or [text]:
        if len(paragraph) <= limit:
            chunks.append(paragraph)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > limit:
                for word in sentence.split():
                    candidate = f"{current} {word}".strip()
                    if len(candidate) <= limit:
                        current = candidate
                    else:
                        if current:
                            chunks.append(current)
                        current = word
                continue
            candidate = f"{current} {sentence}".strip()
            if len(candidate) <= limit:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = sentence
        if current:
            chunks.append(current)
    return chunks or [text]


def tokenize_word_spans(text: str) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for match in re.finditer(r"\b[\w']+\b", text):
        spans.append({"text": match.group(0), "char_start": match.start(), "char_end": match.end()})
    return spans


def assign_word_timings_to_text(text: str, timings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spans = tokenize_word_spans(text)
    for span, timing in zip(spans, timings):
        span["start_time"] = timing["start_time"]
        span["end_time"] = timing["end_time"]
    return spans


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "untitled-track"


def derive_paste_title(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9']+", text)
    if not words:
        return "Untitled Track"
    short_title = " ".join(words[:6]).strip()
    if len(words) > 6:
        short_title += "..."
    return short_title[:80]


def unique_path(directory: Path, stem: str, suffix: str) -> Path:
    candidate = directory / f"{stem}{suffix}"
    counter = 2
    while candidate.exists():
        candidate = directory / f"{stem}-{counter}{suffix}"
        counter += 1
    return candidate


def save_job_metadata(job: Job) -> None:
    metadata_path = JOBS_DIR / f"{job.id}.json"
    metadata_path.write_text(json.dumps(asdict(job), indent=2), encoding="utf-8")


def restore_jobs_from_disk() -> None:
    restored: dict[str, Job] = {}
    for metadata_path in sorted(JOBS_DIR.glob("*.json")):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            job = Job(**payload)
        except Exception:
            continue

        if job.audio_path and not Path(job.audio_path).exists():
            job.state = "failed"
            job.error = "Audio file missing from disk."
        elif job.state in {"queued", "processing"}:
            job.state = "failed"
            job.error = "Processing was interrupted in a previous session."

        restored[job.id] = job

    with jobs_lock:
        jobs.clear()
        jobs.update(restored)


def extract_pdf_text(content: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_pdf:
        temp_pdf.write(content)
        temp_pdf.flush()
        reader = PdfReader(temp_pdf.name)
        extracted_pages: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                extracted_pages.append(page_text.strip())
    return "\n\n".join(extracted_pages)


def extract_pdf_text_with_ocr(content: bytes) -> str:
    ocr = get_ocr_engine()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        pdf_path = temp_dir_path / "upload.pdf"
        pdf_path.write_bytes(content)
        image_prefix = temp_dir_path / "page"

        try:
            subprocess.run(
                ["pdftoppm", "-png", str(pdf_path), str(image_prefix)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Unable to render this PDF for OCR: {exc.stderr.strip()}",
            ) from exc

        page_texts: list[str] = []
        for image_path in sorted(temp_dir_path.glob("page-*.png")):
            result, _ = ocr(str(image_path))
            if not result:
                continue
            page_lines = [item[1].strip() for item in result if len(item) > 1 and item[1].strip()]
            if page_lines:
                page_texts.append("\n".join(page_lines))

    return "\n\n".join(page_texts)


def extract_epub_text(content: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".epub") as temp_epub:
        temp_epub.write(content)
        temp_epub.flush()
        book = epub.read_epub(temp_epub.name)
        sections: list[str] = []
        for item in book.get_items_of_type(ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_body_content(), "lxml")
            text = soup.get_text(separator=" ", strip=True)
            if text:
                sections.append(text)
    return "\n\n".join(sections)


def read_text_file(upload: UploadFile, content: bytes) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix == ".pdf":
        extracted_text = extract_pdf_text(content)
        if not extracted_text.strip():
            extracted_text = extract_pdf_text_with_ocr(content)
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract readable text from this PDF.")
        return extracted_text

    if suffix == ".epub":
        extracted_text = extract_epub_text(content)
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract readable text from this EPUB.")
        return extracted_text

    if suffix not in {".txt", ".md", ".csv", ".log"}:
        raise HTTPException(status_code=400, detail="Supported files are EPUB, PDF, TXT, MD, CSV, and LOG.")
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise HTTPException(status_code=400, detail="The uploaded text file must be UTF-8 text.")


def set_job(job_id: str, **updates: Any) -> None:
    with jobs_lock:
        job = jobs[job_id]
        for key, value in updates.items():
            setattr(job, key, value)
        job.updated_at = time.time()
        save_job_metadata(job)


def save_transcript_data(job_id: str, text: str, words: list[dict[str, Any]], timing_mode: str) -> Path:
    transcript_data_path = TRANSCRIPTS_DIR / f"{job_id}.json"
    transcript_data_path.write_text(
        json.dumps({"text": text, "words": words, "timing_mode": timing_mode}),
        encoding="utf-8",
    )
    return transcript_data_path


def join_edge_audio(parts: list[Path], output_path: Path) -> None:
    concat_file = output_path.with_suffix(".txt")
    concat_file.write_text("".join(f"file '{part.resolve().as_posix()}'\n" for part in parts), encoding="utf-8")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(output_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg could not join the generated audio: {exc.stderr.strip()}") from exc
    finally:
        concat_file.unlink(missing_ok=True)


def get_audio_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return round(float(result.stdout.strip()), 2)


async def synthesize_edge_chunk(text: str, voice_name: str, output_path: Path) -> tuple[list[dict[str, Any]], float]:
    communicate = edge_tts.Communicate(text, voice_name, boundary="WordBoundary")
    boundaries: list[dict[str, Any]] = []
    with open(output_path, "wb") as audio_file:
        async for message in communicate.stream():
            if message["type"] == "audio":
                audio_file.write(message["data"])
            elif message["type"] == "WordBoundary":
                boundaries.append(
                    {
                        "text": message["text"],
                        "start_time": float(message["offset"]) / 10_000_000,
                        "end_time": float(message["offset"] + message["duration"]) / 10_000_000,
                    }
                )
    return boundaries, get_audio_duration(output_path)


async def synthesize_with_edge_async(
    job_id: str, text: str, chunks: list[str], voice_name: str, output_path: Path
) -> tuple[float, list[dict[str, Any]], str]:
    with tempfile.TemporaryDirectory(dir=AUDIO_DIR) as temp_dir:
        temp_paths: list[Path] = []
        all_boundaries: list[dict[str, Any]] = []
        cumulative_time = 0.0

        for index, chunk in enumerate(chunks, start=1):
            temp_path = Path(temp_dir) / f"{index:04d}.mp3"
            chunk_boundaries, chunk_duration = await synthesize_edge_chunk(chunk, voice_name, temp_path)
            for boundary in chunk_boundaries:
                all_boundaries.append(
                    {
                        "text": boundary["text"],
                        "start_time": round(boundary["start_time"] + cumulative_time, 3),
                        "end_time": round(boundary["end_time"] + cumulative_time, 3),
                    }
                )
            temp_paths.append(temp_path)
            cumulative_time += chunk_duration
            set_job(job_id, completed_chunks=index, progress=index / len(chunks))

        join_edge_audio(temp_paths, output_path)

    duration_seconds = get_audio_duration(output_path)
    words = assign_word_timings_to_text(text, all_boundaries)
    return duration_seconds, words, "exact"


def synthesize_with_piper(
    job_id: str, text: str, chunks: list[str], voice_name: str, output_path: Path
) -> tuple[float, list[dict[str, Any]], str]:
    voice = get_piper_voice(voice_name)
    audio_parts: list[np.ndarray] = []
    chunk_infos: list[dict[str, Any]] = []
    sample_rate: int | None = None

    for index, chunk in enumerate(chunks, start=1):
        sentence_audio: list[np.ndarray] = []
        for sentence_chunk in voice.synthesize(chunk):
            sample_rate = sentence_chunk.sample_rate
            sentence_audio.append(sentence_chunk.audio_float_array)

        if sentence_audio:
            combined = np.concatenate(sentence_audio)
            audio_parts.append(combined)
            chunk_infos.append({"text": chunk, "samples": len(combined)})
            if index < len(chunks):
                audio_parts.append(np.zeros(int(sample_rate * CHUNK_GAP_SECONDS), dtype=np.float32))

        set_job(job_id, completed_chunks=index, progress=index / len(chunks))

    if not audio_parts or sample_rate is None:
        raise RuntimeError("No audio was generated for this text.")

    final_audio = np.concatenate(audio_parts)
    sf.write(output_path, final_audio, sample_rate, subtype="PCM_16")
    duration_seconds = round(len(final_audio) / sample_rate, 2)

    timed_words: list[dict[str, Any]] = []
    chunk_cursor = 0.0
    spans = tokenize_word_spans(text)
    span_index = 0
    for info in chunk_infos:
        chunk_duration = info["samples"] / sample_rate
        words_in_chunk = tokenize_word_spans(info["text"])
        if not words_in_chunk:
            chunk_cursor += chunk_duration + CHUNK_GAP_SECONDS
            continue
        total_weight = sum(max(len(word["text"]), 1) for word in words_in_chunk)
        local_time = chunk_cursor
        for word in words_in_chunk:
            duration = chunk_duration * (max(len(word["text"]), 1) / total_weight)
            if span_index < len(spans):
                timed_words.append(
                    {
                        "text": spans[span_index]["text"],
                        "char_start": spans[span_index]["char_start"],
                        "char_end": spans[span_index]["char_end"],
                        "start_time": round(local_time, 3),
                        "end_time": round(local_time + duration, 3),
                    }
                )
            local_time += duration
            span_index += 1
        chunk_cursor += chunk_duration + CHUNK_GAP_SECONDS

    return duration_seconds, timed_words, "estimated"


def synthesize_job(job_id: str) -> None:
    with jobs_lock:
        job = jobs[job_id]
        transcript_path = Path(job.transcript_path or "")
        voice_id = job.voice

    voice_config = get_voice_config(voice_id)
    text = transcript_path.read_text(encoding="utf-8")
    chunks = split_text(text)
    set_job(job_id, state="processing", total_chunks=len(chunks), progress=0.02)

    output_ext = voice_config["format"]
    library_name = job.library_name or slugify(job.title)
    output_path = unique_path(AUDIO_DIR, library_name, f".{output_ext}") if not job.audio_path else Path(job.audio_path)

    if voice_config["provider"] == "edge":
        duration_seconds, words, timing_mode = asyncio.run(
            synthesize_with_edge_async(job_id, text, chunks, voice_config["voice_name"], output_path)
        )
    else:
        duration_seconds, words, timing_mode = synthesize_with_piper(job_id, text, chunks, voice_config["voice_name"], output_path)

    transcript_data_path = save_transcript_data(job_id, text, words, timing_mode)
    set_job(
        job_id,
        state="completed",
        progress=1.0,
        audio_url=f"/audio/{output_path.name}",
        download_url=f"/api/jobs/{job_id}/download",
        transcript_url=f"/api/jobs/{job_id}/transcript",
        duration_seconds=duration_seconds,
        audio_format=output_ext,
        timing_mode=timing_mode,
        transcript_data_path=str(transcript_data_path),
        audio_path=str(output_path),
        library_name=output_path.stem,
    )


def worker() -> None:
    while True:
        job_id = job_queue.get()
        try:
            synthesize_job(job_id)
        except Exception as exc:  # pragma: no cover
            set_job(job_id, state="failed", error=str(exc))
        finally:
            job_queue.task_done()


def queue_job(title: str, source_type: str, text: str, voice_id: str, filename: str | None) -> Job:
    voice_config = get_voice_config(voice_id)
    job_id = uuid.uuid4().hex
    safe_stem = slugify(title)
    transcript_path = unique_path(UPLOADS_DIR, safe_stem, ".txt")
    transcript_path.write_text(text, encoding="utf-8")
    preview = text[:180].replace("\n", " ")
    job = Job(
        id=job_id,
        title=title,
        source_type=source_type,
        voice=voice_id,
        provider=voice_config["provider"],
        voice_label=voice_config["label"],
        state="queued",
        text_length=len(text),
        created_at=time.time(),
        updated_at=time.time(),
        original_filename=filename,
        audio_format=voice_config["format"],
        preview=preview,
        transcript_path=str(transcript_path),
        library_name=safe_stem,
    )
    with jobs_lock:
        jobs[job_id] = job
        save_job_metadata(job)
    job_queue.put(job_id)
    return job


@app.on_event("startup")
def startup() -> None:
    global worker_started
    ensure_piper_voice("en_US-ryan-high")
    ensure_piper_voice("en_US-lessac-medium")
    restore_jobs_from_disk()
    if not worker_started:
        thread = threading.Thread(target=worker, daemon=True, name="tts-worker")
        thread.start()
        worker_started = True


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/voices")
def list_voices() -> dict[str, list[dict[str, str]]]:
    return {
        "voices": [
            {
                "id": voice_id,
                "label": config["label"],
                "provider": config["provider"],
                "description": config["description"],
                "format": config["format"],
            }
            for voice_id, config in VOICE_CATALOG.items()
        ]
    }


@app.get("/api/jobs")
def list_jobs() -> dict[str, list[dict[str, Any]]]:
    with jobs_lock:
        ordered_jobs = sorted(jobs.values(), key=lambda item: item.created_at, reverse=True)
        return {"jobs": [job.payload() for job in ordered_jobs]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return job.payload()


@app.get("/api/jobs/{job_id}/transcript")
def get_transcript(job_id: str) -> dict[str, Any]:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None or not job.transcript_data_path:
            raise HTTPException(status_code=404, detail="Transcript data not found.")
        transcript_data_path = Path(job.transcript_data_path)
    return json.loads(transcript_data_path.read_text(encoding="utf-8"))


@app.get("/api/jobs/{job_id}/download")
def download_audio(job_id: str) -> FileResponse:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None or not job.audio_path or not job.audio_format:
            raise HTTPException(status_code=404, detail="Audio file not found.")
        audio_path = Path(job.audio_path)
        title = slugify(job.title)
        extension = job.audio_format

    media_type = "audio/mpeg" if extension == "mp3" else "audio/wav"
    return FileResponse(audio_path, media_type=media_type, filename=f"{title}.{extension}")


@app.post("/api/jobs")
async def create_job(
    text: str = Form(default=""),
    title: str = Form(default=""),
    voice: str = Form(default=DEFAULT_VOICE),
    file: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    voice_id = voice or DEFAULT_VOICE
    filename: str | None = None
    source_type = "paste"
    file_text = ""

    if file is not None:
        source_type = "file"
        filename = file.filename or "uploaded.txt"
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="Files must be smaller than 5 MB.")
        file_text = read_text_file(file, content)

    merged_text = clean_text(file_text or text)
    if not merged_text:
        raise HTTPException(status_code=400, detail="Paste text or upload a text file.")

    if filename:
        resolved_title = Path(filename).stem
    else:
        resolved_title = clean_text(title) or derive_paste_title(merged_text)

    job = queue_job(resolved_title, source_type, merged_text, voice_id, filename)
    return {"job": job.payload()}
