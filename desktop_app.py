from __future__ import annotations

import json
import multiprocessing
import socket
import time
from contextlib import closing
from urllib.error import URLError
from urllib.request import urlopen

import uvicorn

try:
    import webview
except ImportError as exc:  # pragma: no cover - depends on optional desktop dependency
    raise SystemExit(
        "pywebview is not installed yet. Install dependencies from requirements.txt to use the desktop launcher."
    ) from exc

from app_logging import configure_logging
from app_paths import get_runtime_paths


WINDOW_TITLE = "Rayline Echo"
WINDOW_SIZE = (1440, 960)
WINDOW_MIN_SIZE = (1160, 760)
PATHS = get_runtime_paths()
launcher_logger = configure_logging(PATHS.logs_dir / "desktop-launcher.log", "rayline.desktop")
server_logger = configure_logging(PATHS.logs_dir / "desktop-server.log", "rayline.desktop.server")
LOGO_PATH = (PATHS.static_dir / "branding" / "rayline_echo_wide.png").resolve().as_uri()


def startup_html(*, status_lines: list[str], detail: str | None = None, error: str | None = None) -> str:
    status_items = "".join(f"<li>{line}</li>" for line in status_lines)
    detail_html = f"<p class='detail'>{detail}</p>" if detail else ""
    error_html = f"<p class='error'>{error}</p>" if error else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{WINDOW_TITLE}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #10141f;
      --muted: #5b6477;
      --panel: rgba(255,255,255,0.88);
      --line: rgba(56,88,170,0.14);
      --accent: #2d55b6;
      --bg-a: #f7f8fb;
      --bg-b: #eef3ff;
      --error: #9e352d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: "Manrope", system-ui, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top, rgba(115, 146, 255, 0.18), transparent 40%),
        linear-gradient(180deg, var(--bg-b), var(--bg-a));
      display: grid;
      place-items: center;
      padding: 2rem;
    }}
    .panel {{
      width: min(920px, 100%);
      padding: 1.3rem;
      border-radius: 28px;
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: 0 28px 60px rgba(22, 30, 52, 0.14);
    }}
    .logo {{
      width: 100%;
      display: block;
      border-radius: 22px;
      margin-bottom: 1.2rem;
    }}
    .eyebrow {{
      margin: 0 0 0.35rem;
      letter-spacing: 0.15em;
      text-transform: uppercase;
      font-size: 0.8rem;
      font-weight: 800;
      color: var(--accent);
    }}
    h1 {{
      margin: 0;
      font-size: clamp(2rem, 4vw, 3rem);
      line-height: 0.95;
      font-family: "Source Serif 4", Georgia, serif;
      font-weight: 600;
    }}
    .detail {{
      margin: 0.9rem 0 0;
      color: var(--muted);
      font-size: 1rem;
      line-height: 1.6;
      max-width: 48rem;
    }}
    ul {{
      margin: 1.1rem 0 0;
      padding: 0;
      list-style: none;
      display: grid;
      gap: 0.7rem;
    }}
    li {{
      padding: 0.8rem 0.95rem;
      border-radius: 16px;
      background: rgba(255,255,255,0.78);
      border: 1px solid rgba(56, 88, 170, 0.12);
      color: var(--ink);
      font-size: 0.96rem;
    }}
    .error {{
      margin: 1rem 0 0;
      padding: 0.9rem 1rem;
      border-radius: 16px;
      background: rgba(158, 53, 45, 0.09);
      border: 1px solid rgba(158, 53, 45, 0.18);
      color: var(--error);
      line-height: 1.55;
      white-space: pre-wrap;
    }}
  </style>
</head>
<body>
  <main class="panel">
    <img class="logo" src="{LOGO_PATH}" alt="Rayline Echo logo" />
    <p class="eyebrow">Rayline Echo</p>
    <h1>Preparing your library.</h1>
    {detail_html}
    <ul>{status_items}</ul>
    {error_html}
  </main>
</body>
</html>"""


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def wait_for_server(host: str, port: int, timeout: float = 25.0) -> None:
    deadline = time.time() + timeout
    health_url = f"http://{host}:{port}/api/health"
    while time.time() < deadline:
        try:
            with urlopen(health_url, timeout=1.0) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            time.sleep(0.15)
    raise RuntimeError("Rayline Echo could not start its local server in time.")


def fetch_health(host: str, port: int) -> dict:
    with urlopen(f"http://{host}:{port}/api/health", timeout=2.0) as response:
        return json.loads(response.read().decode("utf-8"))


def run_server(host: str, port: int) -> None:
    server_logger.info("Launching embedded server on %s:%s", host, port)
    from main import app

    uvicorn.run(app, host=host, port=port, log_level="warning", access_log=False)


def launch_window(window: webview.Window, host: str, port: int) -> None:
    status_lines = [
        "Preparing local workspace and reading your saved library",
        "Starting the local Rayline Echo service",
        "Checking local voice runtime and machine health",
        "Opening your listening library",
    ]

    def render(detail: str | None = None, error: str | None = None) -> None:
        window.load_html(startup_html(status_lines=status_lines, detail=detail, error=error))

    render("Rayline Echo is setting up its local workspace, voices, and queue so you can jump straight in.")

    ctx = multiprocessing.get_context("spawn")
    server_process = ctx.Process(target=run_server, args=(host, port), daemon=True, name="rayline-echo-server")
    launcher_logger.info("Starting desktop launcher")
    server_process.start()

    try:
        status_lines[0] = "Workspace ready"
        render("Starting the local service and reconnecting your saved library.")
        wait_for_server(host, port)

        health = fetch_health(host, port)
        worker = health.get("worker", {})
        system = health.get("system", {})
        piper = system.get("piper", {})
        detail = (
            f"Local service is ready. Worker {'online' if worker.get('alive') else 'idle'} • "
            f"Queue {worker.get('queue_size', 0)} • Local voices on {piper.get('label', 'CPU')}."
        )
        status_lines[1] = "Local service started"
        status_lines[2] = "Voice runtime and health checks complete"
        render(detail)
        time.sleep(0.35)

        url = f"http://{host}:{port}"
        launcher_logger.info("Desktop server ready at %s", url)
        status_lines[3] = "Opening your library"
        render(detail)
        window.load_url(url)
    except Exception as exc:
        launcher_logger.exception("Desktop launcher failed")
        stop_server(server_process)
        error_text = (
            f"{exc}\n\nLogs:\n"
            f"- {PATHS.logs_dir / 'desktop-launcher.log'}\n"
            f"- {PATHS.logs_dir / 'desktop-server.log'}\n"
            f"- {PATHS.logs_dir / 'server.log'}"
        )
        render(
            "Rayline Echo could not finish launching. The details below should make the next step clearer.",
            error=error_text,
        )
        return

    window.events.closed += lambda: stop_server(server_process)


def stop_server(server_process: multiprocessing.Process) -> None:
    if server_process.is_alive():
        launcher_logger.info("Stopping desktop server process")
        server_process.terminate()
        server_process.join(timeout=5)


def run_desktop() -> None:
    host = "127.0.0.1"
    port = find_free_port()
    window = webview.create_window(
        WINDOW_TITLE,
        html=startup_html(
            status_lines=[
                "Preparing local workspace and reading your saved library",
                "Starting the local Rayline Echo service",
                "Checking local voice runtime and machine health",
                "Opening your listening library",
            ],
            detail="Rayline Echo is setting up its local workspace, voices, and queue so you can jump straight in.",
        ),
        width=WINDOW_SIZE[0],
        height=WINDOW_SIZE[1],
        min_size=WINDOW_MIN_SIZE,
        text_select=True,
    )
    webview.start(launch_window, args=(window, host, port))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_desktop()
