from __future__ import annotations

import asyncio
import ctypes
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
from urllib.request import urlretrieve

import edge_tts
import numpy as np
import onnxruntime
import soundfile as sf
from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from kokoro_onnx import Kokoro
from piper import PiperVoice
from piper.download_voices import download_voice
from pypdf import PdfReader
from rapidocr_onnxruntime import RapidOCR
from monitor import get_cpu_snapshot, get_gpu_snapshot


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
JOBS_DIR = DATA_DIR / "jobs"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
MODELS_DIR = BASE_DIR / "models"
KOKORO_DIR = MODELS_DIR / "kokoro"
MAX_FILE_SIZE = 5 * 1024 * 1024
CHUNK_LIMIT = 350
CHUNK_GAP_SECONDS = 0.18
KOKORO_CHUNK_LIMIT = 180
KOKORO_MIN_CHUNK_LIMIT = 70
KOKORO_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
KOKORO_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
KOKORO_MODEL_PATH = KOKORO_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = KOKORO_DIR / "voices-v1.0.bin"

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
    "kokoro:af_sky": {
        "provider": "kokoro",
        "voice_name": "af_sky",
        "label": "Sky",
        "description": "Most natural local Kokoro female voice",
        "format": "wav",
    },
    "kokoro:af_heart": {
        "provider": "kokoro",
        "voice_name": "af_heart",
        "label": "Heart",
        "description": "Warm and expressive local Kokoro female voice",
        "format": "wav",
    },
    "kokoro:am_michael": {
        "provider": "kokoro",
        "voice_name": "am_michael",
        "label": "Michael",
        "description": "Most natural local Kokoro male voice",
        "format": "wav",
    },
    "kokoro:am_liam": {
        "provider": "kokoro",
        "voice_name": "am_liam",
        "label": "Liam",
        "description": "Clear and polished local Kokoro male voice",
        "format": "wav",
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

for directory in (STATIC_DIR, DATA_DIR, UPLOADS_DIR, AUDIO_DIR, TRANSCRIPTS_DIR, JOBS_DIR, CHECKPOINTS_DIR, MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
KOKORO_DIR.mkdir(parents=True, exist_ok=True)


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
    compute_target: str | None = None
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
    checkpoint_path: str | None = None
    library_name: str | None = None
    favorite: bool = False
    last_played_at: float | None = None
    started_at: float | None = None
    completed_at: float | None = None

    def payload(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("transcript_path", None)
        data.pop("transcript_data_path", None)
        data.pop("audio_path", None)
        data.pop("checkpoint_path", None)
        data["is_recent"] = bool(self.last_played_at and (time.time() - self.last_played_at) <= 7 * 24 * 60 * 60)
        data["resumable"] = bool(
            self.provider == "edge"
            and self.state != "completed"
            and self.checkpoint_path
            and Path(self.checkpoint_path).exists()
        )
        if self.started_at:
            end_time = self.completed_at or time.time()
            data["processing_elapsed_seconds"] = round(max(0.0, end_time - self.started_at), 2)
        else:
            data["processing_elapsed_seconds"] = None
        data["completed_in_seconds"] = round(max(0.0, self.completed_at - self.started_at), 2) if self.started_at and self.completed_at else None
        return data


app = FastAPI(title="Rayline Echo")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

jobs_lock = threading.Lock()
jobs: dict[str, Job] = {}
job_queue: queue.Queue[str] = queue.Queue()
voices_lock = threading.Lock()
piper_voices: dict[str, PiperVoice] = {}
piper_use_cuda: bool | None = None
piper_runtime_note: str | None = None
kokoro_model: Kokoro | None = None
kokoro_use_cuda: bool | None = None
kokoro_runtime_note: str | None = None
worker_started = False
ocr_lock = threading.Lock()
ocr_engine: RapidOCR | None = None


class JobDeletedError(Exception):
    pass


def get_voice_config(voice_id: str) -> dict[str, str]:
    if voice_id not in VOICE_CATALOG:
        raise HTTPException(status_code=400, detail="Unknown voice selection.")
    return VOICE_CATALOG[voice_id]


def get_piper_runtime() -> dict[str, Any]:
    global piper_use_cuda, piper_runtime_note
    if piper_use_cuda is None:
        if hasattr(onnxruntime, "preload_dlls"):
            try:
                onnxruntime.preload_dlls(directory="")
            except Exception:
                pass
        providers = set(onnxruntime.get_available_providers())
        piper_use_cuda = "CUDAExecutionProvider" in providers
        piper_runtime_note = None
        if piper_use_cuda:
            cuda_provider_lib = Path(onnxruntime.__file__).resolve().parent / "capi" / "libonnxruntime_providers_cuda.so"
            try:
                ctypes.CDLL(str(cuda_provider_lib))
            except OSError as exc:
                piper_use_cuda = False
                piper_runtime_note = str(exc)
    return {
        "available_providers": onnxruntime.get_available_providers(),
        "using_cuda": bool(piper_use_cuda),
        "label": "GPU" if piper_use_cuda else "CPU",
        "note": piper_runtime_note,
    }


def ensure_piper_voice(voice_name: str) -> Path:
    model_path = MODELS_DIR / f"{voice_name}.onnx"
    config_path = MODELS_DIR / f"{voice_name}.onnx.json"
    if model_path.exists() and config_path.exists():
        return model_path
    download_voice(voice_name, MODELS_DIR)
    return model_path


def ensure_kokoro_assets() -> tuple[Path, Path]:
    if not KOKORO_MODEL_PATH.exists():
        urlretrieve(KOKORO_MODEL_URL, KOKORO_MODEL_PATH)
    if not KOKORO_VOICES_PATH.exists():
        urlretrieve(KOKORO_VOICES_URL, KOKORO_VOICES_PATH)
    return KOKORO_MODEL_PATH, KOKORO_VOICES_PATH


def get_piper_voice(voice_name: str) -> PiperVoice:
    global piper_use_cuda, piper_runtime_note
    with voices_lock:
        if voice_name not in piper_voices:
            model_path = ensure_piper_voice(voice_name)
            runtime = get_piper_runtime()
            try:
                piper_voices[voice_name] = PiperVoice.load(model_path, use_cuda=runtime["using_cuda"])
            except Exception:
                if runtime["using_cuda"]:
                    piper_voices[voice_name] = PiperVoice.load(model_path, use_cuda=False)
                    piper_use_cuda = False
                    piper_runtime_note = "CUDA runtime was detected but Piper fell back to CPU during model load."
                else:
                    raise
        return piper_voices[voice_name]


def get_kokoro_model() -> Kokoro:
    global kokoro_model
    with voices_lock:
        if kokoro_model is None:
            model_path, voices_path = ensure_kokoro_assets()
            providers: list[Any] = ["CPUExecutionProvider"]
            runtime = get_kokoro_runtime()
            if runtime["using_cuda"]:
                providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"]
            session = onnxruntime.InferenceSession(str(model_path), providers=providers)
            kokoro_model = Kokoro.from_session(session, str(voices_path))
        return kokoro_model


def get_kokoro_runtime() -> dict[str, Any]:
    global kokoro_use_cuda, kokoro_runtime_note
    if kokoro_use_cuda is None:
        if hasattr(onnxruntime, "preload_dlls"):
            try:
                onnxruntime.preload_dlls(directory="")
            except Exception:
                pass
        kokoro_runtime_note = None
        try:
            test_session = onnxruntime.InferenceSession(
                str(KOKORO_MODEL_PATH if KOKORO_MODEL_PATH.exists() else ensure_kokoro_assets()[0]),
                providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"],
            )
            kokoro_use_cuda = "CUDAExecutionProvider" in test_session.get_providers()
        except Exception as exc:
            kokoro_use_cuda = False
            kokoro_runtime_note = str(exc)
    return {
        "using_cuda": bool(kokoro_use_cuda),
        "label": "GPU" if kokoro_use_cuda else "CPU",
        "note": kokoro_runtime_note,
    }


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


def split_chunk_for_retry(text: str, limit: int) -> list[str]:
    pieces = split_text(text, limit=limit)
    if len(pieces) > 1:
        return pieces

    words = text.split()
    if len(words) <= 1:
        return [text]

    midpoint = len(words) // 2
    left = " ".join(words[:midpoint]).strip()
    right = " ".join(words[midpoint:]).strip()
    return [piece for piece in (left, right) if piece]


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


def normalize_boundary_timings(boundaries: list[dict[str, Any]], chunk_duration: float) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    fallback_word_duration = 0.18

    for index, boundary in enumerate(boundaries):
        start_time = float(boundary.get("start_time", 0.0) or 0.0)
        end_value = boundary.get("end_time")
        end_time = float(end_value) if end_value is not None else 0.0

        if end_time <= start_time:
            next_start = None
            for later in boundaries[index + 1:]:
                later_start = later.get("start_time")
                if later_start is not None:
                    next_start = float(later_start)
                    break

            if next_start is not None and next_start > start_time:
                end_time = next_start
            elif chunk_duration > start_time:
                end_time = min(chunk_duration, start_time + fallback_word_duration)
            else:
                end_time = start_time + fallback_word_duration

        normalized.append(
            {
                **boundary,
                "start_time": round(start_time, 3),
                "end_time": round(max(end_time, start_time + 0.01), 3),
            }
        )

    return normalized


def combine_sections(section_entries: list[dict[str, str]]) -> tuple[str, list[dict[str, Any]]]:
    parts: list[str] = []
    sections: list[dict[str, Any]] = []
    cursor = 0

    for index, entry in enumerate(section_entries, start=1):
        title = clean_text(entry.get("title", "")) or f"Section {index}"
        body = clean_text(entry.get("text", ""))
        if not body:
            continue
        if parts:
            parts.append("\n\n")
            cursor += 2

        start = cursor
        parts.append(body)
        cursor += len(body)
        sections.append(
            {
                "title": title[:120],
                "char_start": start,
                "char_end": cursor,
            }
        )

    return "".join(parts), sections


def assign_section_timings(words: list[dict[str, Any]], sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not sections:
        return []

    timed_sections: list[dict[str, Any]] = []
    for section in sections:
        section_words = [
            word for word in words
            if word["char_start"] >= section["char_start"] and word["char_end"] <= section["char_end"]
        ]
        if section_words:
            timed_sections.append(
                {
                    **section,
                    "start_time": section_words[0]["start_time"],
                    "end_time": section_words[-1]["end_time"],
                }
            )
        else:
            timed_sections.append(section)
    return timed_sections


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "untitled-audiobook"


def derive_paste_title(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9']+", text)
    if not words:
        return "Untitled Audiobook"
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


def remove_checkpoint(checkpoint_path: str | None) -> None:
    if not checkpoint_path:
        return
    path = Path(checkpoint_path)
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file():
            child.unlink(missing_ok=True)
        elif child.is_dir():
            child.rmdir()
    path.rmdir()


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
            if job.provider == "edge" and job.checkpoint_path and Path(job.checkpoint_path).exists():
                job.state = "queued"
                job.error = "Resuming interrupted job from saved checkpoint."
            else:
                job.state = "failed"
                job.error = "Processing was interrupted in a previous session."

        restored[job.id] = job

    with jobs_lock:
        jobs.clear()
        jobs.update(restored)


def requeue_pending_jobs() -> None:
    with jobs_lock:
        pending_ids = [job.id for job in jobs.values() if job.state == "queued"]
    for job_id in pending_ids:
        job_queue.put(job_id)


def extract_pdf_sections(content: bytes) -> list[dict[str, str]]:
    with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_pdf:
        temp_pdf.write(content)
        temp_pdf.flush()
        reader = PdfReader(temp_pdf.name)
        extracted_pages: list[dict[str, str]] = []
        for page_index, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                lines = [line.strip() for line in page_text.splitlines() if line.strip()]
                title = lines[0][:80] if lines else f"Page {page_index}"
                extracted_pages.append({"title": title or f"Page {page_index}", "text": page_text.strip()})
    return extracted_pages


def extract_pdf_sections_with_ocr(content: bytes) -> list[dict[str, str]]:
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

        page_texts: list[dict[str, str]] = []
        for page_index, image_path in enumerate(sorted(temp_dir_path.glob("page-*.png")), start=1):
            result, _ = ocr(str(image_path))
            if not result:
                continue
            page_lines = [item[1].strip() for item in result if len(item) > 1 and item[1].strip()]
            if page_lines:
                page_texts.append(
                    {
                        "title": page_lines[0][:80] or f"Page {page_index}",
                        "text": "\n".join(page_lines),
                    }
                )

    return page_texts


def extract_epub_sections(content: bytes) -> list[dict[str, str]]:
    with tempfile.NamedTemporaryFile(suffix=".epub") as temp_epub:
        temp_epub.write(content)
        temp_epub.flush()
        book = epub.read_epub(temp_epub.name)
        sections: list[dict[str, str]] = []
        for item in book.get_items_of_type(ITEM_DOCUMENT):
            item_name = item.get_name()
            item_stem = Path(item_name).stem.lower()
            if item_stem in {"nav", "toc", "contents", "content"}:
                continue
            soup = BeautifulSoup(item.get_body_content(), "lxml")
            if soup.find(attrs={"epub:type": "toc"}) or soup.find(attrs={"role": "doc-toc"}):
                continue
            heading = soup.find(["h1", "h2", "h3", "title"])
            text = soup.get_text(separator=" ", strip=True)
            if text:
                title = ""
                if heading:
                    title = heading.get_text(" ", strip=True)
                if not title:
                    title = item_name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
                sections.append({"title": title[:120], "text": text})
    return sections


def extract_text_and_sections(upload: UploadFile, content: bytes) -> tuple[str, list[dict[str, Any]]]:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix == ".pdf":
        extracted_sections = extract_pdf_sections(content)
        if not extracted_sections:
            extracted_sections = extract_pdf_sections_with_ocr(content)
        text, sections = combine_sections(extracted_sections)
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract readable text from this PDF.")
        return text, sections

    if suffix == ".epub":
        extracted_sections = extract_epub_sections(content)
        text, sections = combine_sections(extracted_sections)
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract readable text from this EPUB.")
        return text, sections

    if suffix not in {".txt", ".md", ".csv", ".log"}:
        raise HTTPException(status_code=400, detail="Supported files are EPUB, PDF, TXT, MD, CSV, and LOG.")

    for encoding in ("utf-8", "utf-8-sig"):
        try:
            text = content.decode(encoding)
            break
        except UnicodeDecodeError:
            text = ""
            continue
    else:
        raise HTTPException(status_code=400, detail="The uploaded text file must be UTF-8 text.")

    clean = clean_text(text)
    return clean, [{"title": Path(upload.filename or "Text").stem or "Text", "text": clean}]

def set_job(job_id: str, **updates: Any) -> None:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise JobDeletedError(job_id)
        for key, value in updates.items():
            setattr(job, key, value)
        job.updated_at = time.time()
        save_job_metadata(job)


def job_exists(job_id: str) -> bool:
    with jobs_lock:
        return job_id in jobs


def save_transcript_data(
    job_id: str,
    text: str,
    words: list[dict[str, Any]],
    timing_mode: str,
    sections: list[dict[str, Any]],
) -> Path:
    transcript_data_path = TRANSCRIPTS_DIR / f"{job_id}.json"
    transcript_data_path.write_text(
        json.dumps({"text": text, "words": words, "timing_mode": timing_mode, "sections": sections}),
        encoding="utf-8",
    )
    return transcript_data_path


def normalize_sections(text: str, sections: list[dict[str, Any]], fallback_title: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for section in sections or []:
        entry = dict(section)
        if "char_start" not in entry or "char_end" not in entry:
            text_value = clean_text(entry.get("text", ""))
            start_index = text.find(text_value) if text_value else -1
            if start_index >= 0:
                entry["char_start"] = start_index
                entry["char_end"] = start_index + len(text_value)
            else:
                entry["char_start"] = 0
                entry["char_end"] = len(text)
        entry["title"] = clean_text(entry.get("title", "")) or fallback_title
        entry.pop("text", None)
        normalized.append(entry)
    if normalized:
        return normalized
    return [{"title": fallback_title, "char_start": 0, "char_end": len(text)}]


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


def checkpoint_parts_dir(checkpoint_dir: Path) -> Path:
    parts_dir = checkpoint_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    return parts_dir


def checkpoint_meta_dir(checkpoint_dir: Path) -> Path:
    meta_dir = checkpoint_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    return meta_dir


def save_edge_checkpoint_chunk(checkpoint_dir: Path, index: int, chunk_duration: float, boundaries: list[dict[str, Any]]) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    meta_path = checkpoint_meta_dir(checkpoint_dir) / f"{index:04d}.json"
    meta_path.write_text(
        json.dumps({"duration": chunk_duration, "boundaries": boundaries}, ensure_ascii=True),
        encoding="utf-8",
    )


def load_edge_checkpoint(checkpoint_dir: Path, total_chunks: int) -> tuple[int, float, list[dict[str, Any]], list[Path]]:
    if not checkpoint_dir.exists():
        return 0, 0.0, [], []

    parts_dir = checkpoint_dir / "parts"
    meta_dir = checkpoint_dir / "meta"
    if not parts_dir.exists() or not meta_dir.exists():
        return 0, 0.0, [], []

    all_boundaries: list[dict[str, Any]] = []
    temp_paths: list[Path] = []
    cumulative_time = 0.0
    completed_chunks = 0

    for index in range(1, total_chunks + 1):
        part_path = parts_dir / f"{index:04d}.mp3"
        meta_path = meta_dir / f"{index:04d}.json"
        if not part_path.exists() or not meta_path.exists():
            break
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        chunk_duration = float(metadata.get("duration", 0.0) or 0.0)
        boundaries = metadata.get("boundaries", [])
        for boundary in boundaries:
            all_boundaries.append(
                {
                    "text": boundary["text"],
                    "start_time": round(float(boundary["start_time"]) + cumulative_time, 3),
                    "end_time": round(float(boundary["end_time"]) + cumulative_time, 3),
                }
            )
        temp_paths.append(part_path)
        cumulative_time += chunk_duration
        completed_chunks = index

    return completed_chunks, cumulative_time, all_boundaries, temp_paths


async def synthesize_edge_chunk(text: str, voice_name: str, output_path: Path) -> tuple[list[dict[str, Any]], float]:
    communicate = edge_tts.Communicate(text, voice_name, boundary="WordBoundary")
    boundaries: list[dict[str, Any]] = []
    with open(output_path, "wb") as audio_file:
        async for message in communicate.stream():
            if message["type"] == "audio":
                audio_file.write(message["data"])
            elif message["type"] == "WordBoundary":
                start_seconds = float(message["offset"]) / 10_000_000
                duration_seconds = float(message.get("duration", 0.0) or 0.0) / 10_000_000
                boundaries.append(
                    {
                        "text": message["text"],
                        "start_time": start_seconds,
                        "end_time": start_seconds + duration_seconds if duration_seconds > 0 else None,
                    }
                )
    chunk_duration = get_audio_duration(output_path)
    return normalize_boundary_timings(boundaries, chunk_duration), chunk_duration


async def synthesize_with_edge_async(
    job_id: str, text: str, chunks: list[str], voice_name: str, output_path: Path, checkpoint_dir: Path
) -> tuple[float, list[dict[str, Any]], str]:
    parts_dir = checkpoint_parts_dir(checkpoint_dir)
    completed_chunks, cumulative_time, all_boundaries, temp_paths = load_edge_checkpoint(checkpoint_dir, len(chunks))

    if completed_chunks:
        set_job(job_id, completed_chunks=completed_chunks, progress=completed_chunks / len(chunks))

    for index, chunk in enumerate(chunks, start=1):
        if index <= completed_chunks:
            continue
        temp_path = parts_dir / f"{index:04d}.mp3"
        chunk_boundaries, chunk_duration = await synthesize_edge_chunk(chunk, voice_name, temp_path)
        save_edge_checkpoint_chunk(checkpoint_dir, index, chunk_duration, chunk_boundaries)
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
    timed_words = build_estimated_word_timings(text, chunk_infos, sample_rate)
    return duration_seconds, timed_words, "estimated"


def build_estimated_word_timings(text: str, chunk_infos: list[dict[str, Any]], sample_rate: int) -> list[dict[str, Any]]:
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
    return timed_words


def synthesize_kokoro_chunk_safe(
    model: Kokoro,
    text: str,
    voice_name: str,
    *,
    limit: int = KOKORO_CHUNK_LIMIT,
) -> tuple[list[np.ndarray], list[dict[str, Any]], int]:
    try:
        audio, sample_rate = model.create(text, voice=voice_name, lang="en-us")
        audio = audio.astype(np.float32)
        return [audio], [{"text": text, "samples": len(audio)}], sample_rate
    except Exception as exc:
        message = str(exc)
        is_memory_error = "Failed to allocate memory" in message or "RUNTIME_EXCEPTION" in message
        if not is_memory_error or len(text) <= KOKORO_MIN_CHUNK_LIMIT:
            raise

        next_limit = max(KOKORO_MIN_CHUNK_LIMIT, min(limit - 20, len(text) // 2))
        smaller_chunks = split_chunk_for_retry(text, next_limit)
        if len(smaller_chunks) <= 1:
            raise

        audio_parts: list[np.ndarray] = []
        chunk_infos: list[dict[str, Any]] = []
        sample_rate: int | None = None
        for index, smaller_chunk in enumerate(smaller_chunks, start=1):
            sub_parts, sub_infos, sub_rate = synthesize_kokoro_chunk_safe(
                model,
                smaller_chunk,
                voice_name,
                limit=next_limit,
            )
            if sample_rate is None:
                sample_rate = sub_rate
            audio_parts.extend(sub_parts)
            chunk_infos.extend(sub_infos)
            if index < len(smaller_chunks) and sample_rate is not None:
                audio_parts.append(np.zeros(int(sample_rate * CHUNK_GAP_SECONDS), dtype=np.float32))

        if sample_rate is None:
            raise RuntimeError("No audio was generated for this text.")
        return audio_parts, chunk_infos, sample_rate


def synthesize_with_kokoro(
    job_id: str, text: str, chunks: list[str], voice_name: str, output_path: Path
) -> tuple[float, list[dict[str, Any]], str]:
    model = get_kokoro_model()
    audio_parts: list[np.ndarray] = []
    chunk_infos: list[dict[str, Any]] = []
    sample_rate: int | None = None

    for index, chunk in enumerate(chunks, start=1):
        sub_parts, sub_infos, sample_rate = synthesize_kokoro_chunk_safe(model, chunk, voice_name)
        if sub_parts:
            audio_parts.extend(sub_parts)
            chunk_infos.extend(sub_infos)
            if index < len(chunks):
                audio_parts.append(np.zeros(int(sample_rate * CHUNK_GAP_SECONDS), dtype=np.float32))

        set_job(job_id, completed_chunks=index, progress=index / len(chunks))

    if not audio_parts or sample_rate is None:
        raise RuntimeError("No audio was generated for this text.")

    final_audio = np.concatenate(audio_parts)
    sf.write(output_path, final_audio, sample_rate, subtype="PCM_16")
    duration_seconds = round(len(final_audio) / sample_rate, 2)
    timed_words = build_estimated_word_timings(text, chunk_infos, sample_rate)
    return duration_seconds, timed_words, "estimated"


def synthesize_job(job_id: str) -> None:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise JobDeletedError(job_id)
        transcript_path = Path(job.transcript_path or "")
        transcript_data_path = Path(job.transcript_data_path or "") if job.transcript_data_path else None
        voice_id = job.voice
        checkpoint_dir = Path(job.checkpoint_path) if job.checkpoint_path else CHECKPOINTS_DIR / job_id

    voice_config = get_voice_config(voice_id)
    transcript_source = json.loads(transcript_data_path.read_text(encoding="utf-8")) if transcript_data_path and transcript_data_path.exists() else {}
    text = transcript_path.read_text(encoding="utf-8")
    source_sections = transcript_source.get("sections", [])
    chunk_limit = KOKORO_CHUNK_LIMIT if voice_config["provider"] == "kokoro" else CHUNK_LIMIT
    chunks = split_text(text, limit=chunk_limit)
    set_job(job_id, state="processing", total_chunks=len(chunks), progress=0.02, started_at=time.time(), completed_at=None, error=None)

    output_ext = voice_config["format"]
    library_name = job.library_name or slugify(job.title)
    output_path = unique_path(AUDIO_DIR, library_name, f".{output_ext}") if not job.audio_path else Path(job.audio_path)

    if voice_config["provider"] == "edge":
        duration_seconds, words, timing_mode = asyncio.run(
            synthesize_with_edge_async(job_id, text, chunks, voice_config["voice_name"], output_path, checkpoint_dir)
        )
    elif voice_config["provider"] == "kokoro":
        duration_seconds, words, timing_mode = synthesize_with_kokoro(job_id, text, chunks, voice_config["voice_name"], output_path)
    else:
        duration_seconds, words, timing_mode = synthesize_with_piper(job_id, text, chunks, voice_config["voice_name"], output_path)

    if not job_exists(job_id):
        output_path.unlink(missing_ok=True)
        raise JobDeletedError(job_id)

    timed_sections = assign_section_timings(words, source_sections)
    transcript_data_path = save_transcript_data(job_id, text, words, timing_mode, timed_sections)
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
        completed_at=time.time(),
    )
    remove_checkpoint(str(checkpoint_dir))


def worker() -> None:
    while True:
        job_id = job_queue.get()
        try:
            synthesize_job(job_id)
        except JobDeletedError:
            pass
        except Exception as exc:  # pragma: no cover
            if job_exists(job_id):
                try:
                    set_job(job_id, state="failed", error=str(exc), completed_at=time.time())
                except JobDeletedError:
                    pass
        finally:
            job_queue.task_done()


def queue_job(
    title: str,
    source_type: str,
    text: str,
    voice_id: str,
    filename: str | None,
    *,
    job_id: str | None = None,
    transcript_data_path: str | None = None,
    favorite: bool = False,
) -> Job:
    voice_config = get_voice_config(voice_id)
    if voice_config["provider"] == "edge":
        compute_target = "cloud"
    elif voice_config["provider"] == "kokoro":
        compute_target = get_kokoro_runtime()["label"].lower()
    else:
        compute_target = get_piper_runtime()["label"].lower()
    resolved_job_id = job_id or uuid.uuid4().hex
    safe_stem = slugify(title)
    transcript_path = unique_path(UPLOADS_DIR, safe_stem, ".txt")
    transcript_path.write_text(text, encoding="utf-8")
    preview = text[:180].replace("\n", " ")
    checkpoint_path = str(CHECKPOINTS_DIR / resolved_job_id) if voice_config["provider"] == "edge" else None
    job = Job(
        id=resolved_job_id,
        title=title,
        source_type=source_type,
        voice=voice_id,
        provider=voice_config["provider"],
        voice_label=voice_config["label"],
        compute_target=compute_target,
        state="queued",
        text_length=len(text),
        created_at=time.time(),
        updated_at=time.time(),
        original_filename=filename,
        audio_format=voice_config["format"],
        preview=preview,
        transcript_path=str(transcript_path),
        transcript_data_path=transcript_data_path,
        checkpoint_path=checkpoint_path,
        library_name=safe_stem,
        favorite=favorite,
    )
    with jobs_lock:
        jobs[resolved_job_id] = job
        save_job_metadata(job)
    job_queue.put(resolved_job_id)
    return job


def create_job_from_text(
    *,
    text: str,
    title: str,
    voice_id: str,
    source_type: str,
    filename: str | None,
    sections: list[dict[str, Any]] | None = None,
    favorite: bool = False,
) -> Job:
    resolved_title = clean_text(title) or derive_paste_title(text)
    normalized_sections = normalize_sections(text, sections or [], resolved_title)
    job_id = uuid.uuid4().hex
    source_transcript_path = TRANSCRIPTS_DIR / f"{job_id}.json"
    source_transcript_path.write_text(
        json.dumps({"text": text, "words": [], "timing_mode": "pending", "sections": normalized_sections}),
        encoding="utf-8",
    )
    return queue_job(
        resolved_title,
        source_type,
        text,
        voice_id,
        filename,
        job_id=job_id,
        transcript_data_path=str(source_transcript_path),
        favorite=favorite,
    )


@app.on_event("startup")
def startup() -> None:
    global worker_started
    ensure_piper_voice("en_US-ryan-high")
    ensure_piper_voice("en_US-lessac-medium")
    restore_jobs_from_disk()
    requeue_pending_jobs()
    if not worker_started:
        thread = threading.Thread(target=worker, daemon=True, name="tts-worker")
        thread.start()
        worker_started = True


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/voices", response_model=None)
def list_voices() -> dict[str, Any]:
    piper_runtime = get_piper_runtime()
    kokoro_runtime = get_kokoro_runtime()
    return {
        "voices": [
            {
                "id": voice_id,
                "label": config["label"],
                "provider": config["provider"],
                "description": config["description"],
                "format": config["format"],
                "compute_target": (
                    "cloud"
                    if config["provider"] == "edge"
                    else kokoro_runtime["label"].lower() if config["provider"] == "kokoro"
                    else piper_runtime["label"].lower()
                ),
                "compute_label": (
                    "Microsoft cloud"
                    if config["provider"] == "edge"
                    else f"Local {kokoro_runtime['label']}" if config["provider"] == "kokoro"
                    else f"Local {piper_runtime['label']}"
                ),
                "compute_note": (
                    ""
                    if config["provider"] == "edge"
                    else (kokoro_runtime.get("note") or "") if config["provider"] == "kokoro"
                    else (piper_runtime.get("note") or "")
                ),
            }
            for voice_id, config in VOICE_CATALOG.items()
        ]
    }


@app.get("/api/jobs", response_model=None)
def list_jobs() -> dict[str, Any]:
    with jobs_lock:
        ordered_jobs = sorted(
            jobs.values(),
            key=lambda item: (
                not item.favorite,
                -(item.last_played_at or 0),
                -item.created_at,
            ),
        )
        payload = {"jobs": [job.payload() for job in ordered_jobs]}
    try:
        payload["system"] = system_status()
    except Exception:
        payload["system"] = {
            "timestamp": time.time(),
            "cpu": {"available": False, "summary": "CPU stats unavailable right now."},
            "gpu": {"available": False, "summary": "GPU stats unavailable right now."},
        }
    return payload


@app.get("/api/system")
def system_status() -> dict[str, Any]:
    try:
        cpu = get_cpu_snapshot()
    except Exception:
        cpu = {"available": False, "summary": "CPU stats unavailable right now."}

    try:
        gpu = get_gpu_snapshot()
    except Exception:
        gpu = {"available": False, "summary": "GPU stats unavailable right now."}

    return {
        "timestamp": time.time(),
        "cpu": cpu,
        "gpu": gpu,
        "piper": get_piper_runtime(),
    }


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


@app.patch("/api/jobs/{job_id}")
def update_job(
    job_id: str,
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if "title" in payload:
        title = clean_text(str(payload.get("title", "")))
        if not title:
            raise HTTPException(status_code=400, detail="Title cannot be empty.")
        updates["title"] = title[:160]
    if "favorite" in payload:
        updates["favorite"] = bool(payload.get("favorite"))
    if "touch_recent" in payload and payload.get("touch_recent"):
        updates["last_played_at"] = time.time()

    if not updates:
        raise HTTPException(status_code=400, detail="No valid updates were provided.")

    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        for key, value in updates.items():
            setattr(job, key, value)
        job.updated_at = time.time()
        save_job_metadata(job)
        return {"job": job.payload()}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict[str, bool]:
    with jobs_lock:
        job = jobs.pop(job_id, None)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        metadata_path = JOBS_DIR / f"{job_id}.json"

    for path_value in (job.audio_path, job.transcript_path, job.transcript_data_path, str(metadata_path)):
        if not path_value:
            continue
        try:
            Path(path_value).unlink(missing_ok=True)
        except OSError:
            continue
    remove_checkpoint(job.checkpoint_path)

    return {"deleted": True}


@app.post("/api/jobs/{job_id}/resume")
def resume_job(job_id: str) -> dict[str, Any]:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        if job.provider != "edge" or not job.checkpoint_path or not Path(job.checkpoint_path).exists():
            raise HTTPException(status_code=400, detail="This job does not have resumable Edge checkpoints.")
        if job.state == "completed":
            raise HTTPException(status_code=400, detail="Completed titles do not need resuming.")
        if job.state in {"queued", "processing"}:
            raise HTTPException(status_code=400, detail="This job is already active.")
        job.state = "queued"
        job.error = "Resuming from saved checkpoint."
        job.completed_at = None
        job.updated_at = time.time()
        save_job_metadata(job)
    job_queue.put(job_id)
    return {"job": job.payload()}


@app.post("/api/jobs/{job_id}/reprocess")
def reprocess_job(
    job_id: str,
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    voice_id = str(payload.get("voice") or "").strip()
    if not voice_id:
        raise HTTPException(status_code=400, detail="Choose a voice to create another version.")
    get_voice_config(voice_id)

    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        transcript_path = Path(job.transcript_path or "")
        transcript_data_path = Path(job.transcript_data_path or "") if job.transcript_data_path else None
        source_title = job.title
        source_type = job.source_type
        source_filename = job.original_filename
        source_favorite = job.favorite

    if not transcript_path.exists():
        raise HTTPException(status_code=404, detail="Original source text is missing for this title.")

    text = transcript_path.read_text(encoding="utf-8")
    sections: list[dict[str, Any]] = []
    if transcript_data_path and transcript_data_path.exists():
        transcript_data = json.loads(transcript_data_path.read_text(encoding="utf-8"))
        sections = transcript_data.get("sections", [])

    new_job = create_job_from_text(
        text=text,
        title=source_title,
        voice_id=voice_id,
        source_type=source_type,
        filename=source_filename,
        sections=sections,
        favorite=source_favorite,
    )
    return {"job": new_job.payload()}


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
    sections: list[dict[str, Any]] = []

    if file is not None:
        suffix = Path(file.filename or "").suffix.lower()
        source_type = suffix[1:] if suffix else "file"
        filename = file.filename or "uploaded.txt"
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="Files must be smaller than 5 MB.")
        file_text, sections = extract_text_and_sections(file, content)

    merged_text = clean_text(file_text or text)
    if not merged_text:
        raise HTTPException(status_code=400, detail="Paste text or upload a reading source.")

    if filename:
        resolved_title = Path(filename).stem
    else:
        resolved_title = clean_text(title) or derive_paste_title(merged_text)

    job = create_job_from_text(
        text=merged_text,
        title=resolved_title,
        voice_id=voice_id,
        source_type=source_type,
        filename=filename,
        sections=sections,
    )
    return {"job": job.payload()}
