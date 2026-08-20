from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image


APP_NAME = "Rayline Echo"
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
BRANDING_DIR = STATIC_DIR / "branding"
BUILD_DIR = BASE_DIR / "build"
DIST_DIR = BASE_DIR / "dist"
ICON_SOURCE = BRANDING_DIR / "rayline_browser_favicon_square.png"
WINDOWS_ICON = BUILD_DIR / "icons" / "rayline_echo.ico"
MAC_ICON = BUILD_DIR / "icons" / "rayline_echo.icns"


def ensure_icon_source() -> None:
    if not ICON_SOURCE.exists():
        raise SystemExit(f"Missing icon source: {ICON_SOURCE}")


def ensure_windows_icon() -> Path:
    ensure_icon_source()
    WINDOWS_ICON.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(ICON_SOURCE).convert("RGBA")
    image.save(
        WINDOWS_ICON,
        format="ICO",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    return WINDOWS_ICON


def ensure_mac_icon() -> Path | None:
    ensure_icon_source()
    MAC_ICON.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(ICON_SOURCE).convert("RGBA")
    try:
        image.save(MAC_ICON, format="ICNS")
        return MAC_ICON
    except Exception:
        return None


def pyinstaller_data_arg(path: Path, target: str) -> str:
    separator = ";" if platform.system() == "Windows" else ":"
    return f"{path}{separator}{target}"


def build() -> int:
    if shutil.which("pyinstaller") is None:
        raise SystemExit(
            "PyInstaller is not installed. Run `python3 -m pip install -r requirements-packaging.txt` first."
        )

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",
        "--name",
        APP_NAME,
        "--add-data",
        pyinstaller_data_arg(STATIC_DIR, "static"),
        "--collect-all",
        "webview",
        "desktop_app.py",
    ]

    system = platform.system()
    if system == "Windows":
        command.extend(["--icon", str(ensure_windows_icon())])
    elif system == "Darwin":
        mac_icon = ensure_mac_icon()
        if mac_icon is not None:
            command.extend(["--icon", str(mac_icon)])

    subprocess.run(command, check=True, cwd=BASE_DIR)
    print()
    print(f"{APP_NAME} desktop build complete.")
    print(f"Bundle location: {DIST_DIR / APP_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
