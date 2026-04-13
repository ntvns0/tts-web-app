# TTS Web App

A local-first text-to-speech web app built with FastAPI, premium Edge neural voices, and Piper.

Created by Nate Evans, with implementation assistance from OpenAI Codex.

## Creator And Usage Terms

- Creator: Nate Evans
- Assisted by: OpenAI Codex
- Personal, educational, and non-commercial use is allowed
- Commercial or for-profit use is not allowed without permission from Nate Evans
- If you want to use this project for profit, contact Nate Evans first to arrange a one-time paid license

By using this repository, you agree to the terms in [LICENSE](LICENSE).

## Features

- Upload an EPUB, PDF, or plain text file, paste text, or drag and drop a file
- Queue TTS jobs and watch chunk-by-chunk processing progress
- Choose between more natural premium neural voices and offline local voices
- Keep completed tracks in a selectable library that survives app restarts
- Play, pause, skip forward/back 5 seconds, and scrub on a timeline
- Follow the spoken text in a synced read-along viewer with word highlighting

## Run

```bash
python3 -m pip install -r requirements.txt
python3 -m uvicorn main:app --reload
```

Open `http://127.0.0.1:8000`.

## Notes

- The app offers higher-quality Microsoft Edge neural voices by default. These require internet access during synthesis.
- The app also keeps local Piper fallback voices in `./models`, including `en_US-ryan-high` and `en_US-lessac-medium`.
- Uploaded files must be under 5 MB. Text files must be UTF-8.
- PDF uploads first use embedded text extraction, then fall back to local OCR for scanned/image PDFs.
- EPUB uploads extract chapter/document text and feed it into the same TTS and synced transcript flow.
- Generated `data/` files and downloaded `models/` are intentionally ignored by git.
- Track metadata is stored locally so completed audio files show back up in the library after you restart the app.
- Uploaded text files use their source filename for the generated audio. Pasted text can use a user-supplied title or an automatic title based on the opening words.

## Licensing

This project is source-available under a custom non-commercial license.

- You may use, study, modify, and share the code for non-commercial purposes
- You may not use this code in any commercial, business, paid, or for-profit setting without permission
- For commercial licensing, contact Nate Evans through GitHub: <https://github.com/ntvns0>
