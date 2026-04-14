# Rayline Echo

Turn documents, books, and notes into a personal audiobook library.

Rayline Echo is a local-first web app for turning written content into a listenable experience. Import PDFs, EPUBs, Markdown, text files, and scanned documents, then build a private library you can queue, revisit, and continue listening to over time.

Created by Nate Evans, with implementation assistance from OpenAI Codex.

## What Rayline Echo Does

Rayline Echo is built for people who want to listen to what they read.

- Turn books, papers, notes, articles, and drafts into long-form audio
- Import PDFs, EPUBs, Markdown, TXT, CSV, and LOG files
- Extract text from scanned or flattened PDFs with local OCR
- Queue larger conversions and follow progress while your library builds
- Keep a persistent audiobook library between sessions
- Listen with playback controls, timeline scrubbing, and playback continuity
- Follow along with synced read-along highlighting while audio plays
- Organize your library with favorites, recents, renaming, filtering, search, and reprocessing
- Use high-quality local voices with GPU acceleration when available
- Optionally use cloud voices when you want a different listening style

## Supported Formats

Rayline Echo currently supports:

- PDF
- EPUB
- Markdown
- TXT
- CSV
- LOG

For PDFs, Rayline Echo first attempts embedded text extraction. If a PDF is scanned, flattened, or image-based, it falls back to local OCR.

## Voice Options

Rayline Echo supports both local and cloud-backed voice workflows.

### Local Voices

Local voices are the best fit for private, long-form listening.

- Kokoro local premium voices
- Piper local fallback voices
- Local GPU acceleration when supported by your system
- Local-first processing for users who want privacy and control

### Cloud Voices

Optional cloud voices are available when you want a different voice profile or quality tradeoff.

- Microsoft Edge neural voices
- Useful when you want another voice style alongside your local library workflow

## Why Rayline Echo Is Different

Rayline Echo is not designed as a generic text-to-speech utility.

It is designed as a listening product for real reading workflows: books, papers, notes, research, drafts, and personal documents. The focus is on turning written material into something you can actually live with over time: queue it, organize it, revisit it, and keep listening.

What makes it different:

- Local-first by design
- Built around long-form listening, not one-off voice generation
- Designed for personal libraries, not disposable conversions
- OCR support for real-world PDFs that are not already machine-readable
- Read-along playback with synchronized highlighting
- Persistent library and queue workflow instead of one-time exports

## Run Rayline Echo

From the project directory:

```bash
python3 -m pip install -r requirements.txt
python3 -m uvicorn main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

To make it available on your local network:

```bash
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## Setup Notes

- Uploaded files must be under 5 MB
- Text-based uploads should be UTF-8
- EPUB and PDF imports preserve sections when possible for easier navigation
- Scanned PDFs use local OCR when embedded text is unavailable
- Long-form local synthesis uses chunked processing and adaptive fallback for better reliability
- Generated data is stored locally and intentionally ignored by git

## Privacy And Local-First Design

Rayline Echo is built to keep your reading and listening workflow close to your own machine.

- Local voices run on your hardware
- OCR runs locally
- Files and generated audio stay in your local app data
- Persistent library metadata is stored locally
- Cloud voice options are optional, not required

If you prefer a fully local workflow, you can stay on local voices and avoid cloud-backed synthesis entirely.

## Monitoring

If you want to watch active conversions and system load from the terminal:

```bash
python3 monitor.py jobs
python3 monitor.py processes --limit 10
python3 monitor.py cpu
python3 monitor.py gpu
python3 monitor.py watch --interval 3
```

`watch` combines queued conversions, top processes, CPU activity, and GPU usage when `nvidia-smi` is available.

## Technical Stack

Rayline Echo is built with:

- FastAPI
- Kokoro ONNX for premium local voices
- Piper for lightweight local fallback voices
- Edge TTS for optional cloud voices
- pypdf for PDF extraction
- RapidOCR for scanned PDF OCR
- ebooklib plus BeautifulSoup for EPUB extraction

## Licensing Summary

Rayline Echo is source-available under a custom non-commercial license.

- Creator: Nate Evans
- Built with assistance from OpenAI Codex
- Personal, educational, and other non-commercial use is allowed
- Commercial or for-profit use requires permission from Nate Evans and a one-time paid license

By using this repository, you agree to the terms in [LICENSE](LICENSE).
