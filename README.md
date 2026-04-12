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

- Upload a plain text file, paste text, or drag and drop a file
- Queue TTS jobs and watch chunk-by-chunk processing progress
- Choose between more natural premium neural voices and offline local voices
- Keep completed tracks in a selectable list
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
- Uploaded files must be UTF-8 plain text and under 5 MB.
- Generated `data/` files and downloaded `models/` are intentionally ignored by git.

## Licensing

This project is source-available under a custom non-commercial license.

- You may use, study, modify, and share the code for non-commercial purposes
- You may not use this code in any commercial, business, paid, or for-profit setting without permission
- For commercial licensing, contact Nate Evans through GitHub: <https://github.com/ntvns0>
