from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

from PIL import Image

from build_desktop import APP_NAME, BASE_DIR, BRANDING_DIR, DIST_DIR, build as build_desktop_bundle


APP_ID = "io.rayline.RaylineEcho"
LINUX_BUILD_DIR = BASE_DIR / "build" / "linux"
APPDIR = LINUX_BUILD_DIR / "RaylineEcho.AppDir"
PYINSTALLER_DIR = DIST_DIR / APP_NAME
APPIMAGE_NAME = "Rayline-Echo-x86_64.AppImage"
ARCHIVE_NAME = "rayline-echo-linux-x86_64.tar.gz"
ICON_SOURCE = BRANDING_DIR / "rayline_browser_favicon_square.png"
DESKTOP_FILENAME = f"{APP_ID}.desktop"
APPDATA_SOURCE = BASE_DIR / "packaging" / "linux" / "rayline-echo.appdata.xml"
APPIMAGETOOL_URL = "https://github.com/AppImage/appimagetool/releases/latest/download/appimagetool-x86_64.AppImage"
TOOLS_DIR = BASE_DIR / "build" / "tools"
APPIMAGETOOL_PATH = TOOLS_DIR / "appimagetool-x86_64.AppImage"


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_desktop_file(target: Path) -> None:
    target.write_text(
        """[Desktop Entry]
Type=Application
Name=Rayline Echo
Comment=Turn documents, books, and notes into a personal audiobook library
Exec=rayline-echo
Icon=rayline-echo
Terminal=false
Categories=AudioVideo;Audio;
Keywords=audiobook;tts;reading;epub;pdf;notes;
StartupWMClass=Rayline Echo
""",
        encoding="utf-8",
    )
    applications_dir = target.parent / "usr" / "share" / "applications"
    applications_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target, applications_dir / DESKTOP_FILENAME)


def write_apprun(target: Path) -> None:
    target.write_text(
        """#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/bin/rayline-echo" "$@"
""",
        encoding="utf-8",
    )
    target.chmod(0o755)


def stage_icon(appdir: Path) -> None:
    icon_dir = appdir / "usr" / "share" / "icons" / "hicolor" / "512x512" / "apps"
    icon_dir.mkdir(parents=True, exist_ok=True)
    icon_path = icon_dir / "rayline-echo.png"
    Image.open(ICON_SOURCE).convert("RGBA").save(icon_path, format="PNG")
    shutil.copy2(icon_path, appdir / "rayline-echo.png")
    shutil.copy2(icon_path, appdir / ".DirIcon")


def stage_appdata(appdir: Path) -> None:
    target_dir = appdir / "usr" / "share" / "metainfo"
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(APPDATA_SOURCE, target_dir / f"{APP_ID}.appdata.xml")


def stage_bundle(appdir: Path) -> None:
    if not PYINSTALLER_DIR.exists():
        raise SystemExit(
            f"Missing desktop bundle at {PYINSTALLER_DIR}. Run `python3 build_desktop.py` first, or let this script build it."
        )

    lib_dir = appdir / "usr" / "lib" / "rayline-echo"
    bin_dir = appdir / "usr" / "bin"
    lib_dir.parent.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(PYINSTALLER_DIR, lib_dir, dirs_exist_ok=True)

    launcher = bin_dir / "rayline-echo"
    launcher.write_text(
        """#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
APP_ROOT="$HERE/../lib/rayline-echo"
export PYTHONUTF8=1
export PYTHONIOENCODING=UTF-8
exec "$APP_ROOT/Rayline Echo" "$@"
""",
        encoding="utf-8",
    )
    launcher.chmod(0o755)


def create_archive(appdir: Path) -> Path:
    archive_path = DIST_DIR / ARCHIVE_NAME
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(appdir, arcname="RaylineEcho")
    return archive_path


def build_appimage(appdir: Path, appimagetool: str) -> Path:
    output_path = DIST_DIR / APPIMAGE_NAME
    if output_path.exists():
        output_path.unlink()
    env = dict(os.environ, ARCH="x86_64")
    subprocess.run([appimagetool, "--appimage-extract-and-run", str(appdir), str(output_path)], check=True, cwd=BASE_DIR, env=env)
    return output_path


def ensure_appimagetool(path_hint: str) -> str | None:
    if path_hint:
        return path_hint

    discovered = shutil.which("appimagetool")
    if discovered:
        return discovered

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    if not APPIMAGETOOL_PATH.exists():
        print(f"Downloading appimagetool to {APPIMAGETOOL_PATH}")
        urlretrieve(APPIMAGETOOL_URL, APPIMAGETOOL_PATH)
        APPIMAGETOOL_PATH.chmod(0o755)
    return str(APPIMAGETOOL_PATH)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Linux release target for Rayline Echo.")
    parser.add_argument("--skip-bundle", action="store_true", help="Use the existing PyInstaller bundle without rebuilding it.")
    parser.add_argument("--appimage-tool", default="", help="Path to appimagetool if available.")
    parser.add_argument("--no-appimage", action="store_true", help="Skip AppImage creation and only build the staged archive.")
    args = parser.parse_args()

    if not args.skip_bundle:
        build_desktop_bundle()

    ensure_clean_dir(APPDIR)
    write_desktop_file(APPDIR / DESKTOP_FILENAME)
    write_apprun(APPDIR / "AppRun")
    stage_icon(APPDIR)
    stage_appdata(APPDIR)
    stage_bundle(APPDIR)

    subprocess.run(["desktop-file-validate", str(APPDIR / DESKTOP_FILENAME)], check=True, cwd=BASE_DIR)

    archive_path = create_archive(APPDIR)
    print(f"Linux release archive ready: {archive_path}")

    appimagetool = None if args.no_appimage else ensure_appimagetool(args.appimage_tool)
    if appimagetool:
        appimage_path = build_appimage(APPDIR, appimagetool)
        print(f"AppImage ready: {appimage_path}")
    else:
        print("AppImage not built.")

    print(f"Staged AppDir: {APPDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
