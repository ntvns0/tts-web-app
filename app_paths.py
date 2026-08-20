from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_data_dir


APP_NAME = "Rayline Echo"
APP_AUTHOR = "Rayline"


@dataclass(frozen=True)
class RuntimePaths:
    app_name: str
    app_author: str
    base_dir: Path
    resource_dir: Path
    static_dir: Path
    app_state_dir: Path
    data_dir: Path
    models_dir: Path
    uploads_dir: Path
    audio_dir: Path
    transcripts_dir: Path
    jobs_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    kokoro_dir: Path
    packaged: bool

    def ensure(self) -> "RuntimePaths":
        for directory in (
            self.static_dir,
            self.app_state_dir,
            self.data_dir,
            self.models_dir,
            self.uploads_dir,
            self.audio_dir,
            self.transcripts_dir,
            self.jobs_dir,
            self.checkpoints_dir,
            self.logs_dir,
            self.kokoro_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self


def get_runtime_paths() -> RuntimePaths:
    base_dir = Path(__file__).resolve().parent
    packaged = bool(getattr(sys, "frozen", False))
    resource_dir = Path(getattr(sys, "_MEIPASS", base_dir))
    static_dir = resource_dir / "static"

    if packaged:
        app_state_dir = Path(user_data_dir(APP_NAME, APP_AUTHOR))
        data_dir = app_state_dir / "data"
        models_dir = app_state_dir / "models"
    else:
        app_state_dir = base_dir
        data_dir = base_dir / "data"
        models_dir = base_dir / "models"

    return RuntimePaths(
        app_name=APP_NAME,
        app_author=APP_AUTHOR,
        base_dir=base_dir,
        resource_dir=resource_dir,
        static_dir=static_dir,
        app_state_dir=app_state_dir,
        data_dir=data_dir,
        models_dir=models_dir,
        uploads_dir=data_dir / "uploads",
        audio_dir=data_dir / "audio",
        transcripts_dir=data_dir / "transcripts",
        jobs_dir=data_dir / "jobs",
        checkpoints_dir=data_dir / "checkpoints",
        logs_dir=app_state_dir / "logs",
        kokoro_dir=models_dir / "kokoro",
        packaged=packaged,
    ).ensure()
