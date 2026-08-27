#!/usr/bin/env python3
"""
Serve the DNA–CaOx viewer and submit Gaussian 16 jobs locally.

Usage (from repo root):
  python3 scripts/viewer_server.py
  open http://localhost:8765/viewer/

Requires Gaussian 16 (or g09) on PATH, G16_COMMAND, or /Applications/g16/g16.
GaussView exports land in viewer/gaussview_exports/.
Jobs run in viewer/g16_jobs/<id>/ with live log tail in the viewer UI.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
JOBS_DIR = ROOT / "viewer" / "g16_jobs"
EXPORTS_DIR = ROOT / "viewer" / "gaussview_exports"
LOG_TAIL_BYTES = 12000
DEFAULT_G16_PATHS = (
    Path("/Applications/g16/g16"),
    Path("/Applications/Gaussian/G16/g16"),
)

SCF_DONE_RE = re.compile(r"SCF Done:\s+E\([^)]+\)\s*=\s*([-0-9.]+)")
NORMAL_RE = re.compile(r"Normal termination", re.I)
ERROR_RE = re.compile(r"Error termination", re.I)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def find_g16() -> str | None:
    override = os.environ.get("G16_COMMAND", "").strip()
    if override:
        p = Path(override).expanduser()
        if p.exists():
            return str(p)
        if shutil.which(override):
            return override
    for cmd in ("g16", "g09"):
        found = shutil.which(cmd)
        if found:
            return found
    for env_key in ("GAUSS_EXEDIR", "g16root", "G16ROOT"):
        root = os.environ.get(env_key)
        if not root:
            continue
        base = Path(root).expanduser()
        for name in ("g16", "g09"):
            candidate = base / name
            if candidate.exists():
                return str(candidate)
    for candidate in DEFAULT_G16_PATHS:
        if candidate.exists():
            return str(candidate)
    return None


G16_CMD = find_g16()


def current_g16_cmd() -> str | None:
    """Re-resolve g16 path (install path may appear after server start)."""
    global G16_CMD
    G16_CMD = find_g16()
    return G16_CMD


def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def status_path(job_id: str) -> Path:
    return job_dir(job_id) / "status.json"


def read_status(job_id: str) -> dict | None:
    path = status_path(job_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_status(job_id: str, data: dict) -> None:
    data["updated"] = utc_now()
    status_path(job_id).write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_log(log_path: Path) -> dict:
    out: dict = {}
    if not log_path.exists():
        return out
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    energies = SCF_DONE_RE.findall(text)
    if energies:
        out["lastEnergy"] = float(energies[-1])
    if NORMAL_RE.search(text):
        out["state"] = "completed"
    elif ERROR_RE.search(text):
        out["state"] = "failed"
    return out


def tail_log(log_path: Path, nbytes: int = LOG_TAIL_BYTES) -> str:
    if not log_path.exists():
        return ""
    try:
        size = log_path.stat().st_size
        with log_path.open("rb") as fh:
            if size > nbytes:
                fh.seek(-nbytes)
            data = fh.read()
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


def list_jobs() -> list[dict]:
    if not JOBS_DIR.exists():
        return []
    jobs = []
    for child in sorted(JOBS_DIR.iterdir(), key=lambda p: p.name, reverse=True):
        if not child.is_dir():
            continue
        st = read_status(child.name)
        if not st:
            continue
        log_path = child / f"{child.name}.log"
        st = dict(st)
        st["id"] = child.name
        st["logTail"] = tail_log(log_path, 2000)
        jobs.append(st)
    return jobs


def run_job(job_id: str) -> None:
    jd = job_dir(job_id)
    com_path = jd / f"{job_id}.com"
    log_path = jd / f"{job_id}.log"
    pid_path = jd / "pid"
    st = read_status(job_id) or {}
    st["state"] = "running"
    write_status(job_id, st)

    g16_cmd = current_g16_cmd()
    if not g16_cmd:
        st["state"] = "failed"
        st["error"] = "Gaussian executable not found (set G16_COMMAND or PATH)."
        write_status(job_id, st)
        return

    env = os.environ.copy()
    scratch = jd / "scratch"
    scratch.mkdir(exist_ok=True)
    env.setdefault("GAUSS_SCRDIR", str(scratch))
    g16root = Path(g16_cmd).resolve().parent
    env.setdefault("g16root", str(g16root))
    env.setdefault("GAUSS_EXEDIR", str(g16root))

    try:
        with log_path.open("w", encoding="utf-8") as logfh:
            proc = subprocess.Popen(
                [g16_cmd, str(com_path)],
                cwd=jd,
                stdout=logfh,
                stderr=subprocess.STDOUT,
                env=env,
            )
        pid_path.write_text(str(proc.pid), encoding="utf-8")
        exit_code = proc.wait()
        pid_path.unlink(missing_ok=True)
    except Exception as exc:
        st = read_status(job_id) or {}
        st["state"] = "failed"
        st["error"] = str(exc)
        write_status(job_id, st)
        return

    st = read_status(job_id) or {}
    parsed = parse_log(log_path)
    st.update(parsed)
    st["exitCode"] = exit_code
    if st.get("state") not in ("completed", "failed"):
        st["state"] = "completed" if exit_code == 0 else "failed"
    if st["state"] == "failed" and not st.get("error"):
        st["error"] = f"Gaussian exited with code {exit_code}"
    write_status(job_id, st)


def safe_export_stem(label: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "-", label.strip())[:80].strip("-")
    return stem or "g16_export"


def write_gaussview_export(com: str, label: str, formats: list[str]) -> dict[str, str]:
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{safe_export_stem(label)}_g16"
    text = com + ("\n" if not com.endswith("\n") else "")
    paths: dict[str, str] = {}
    for fmt in formats:
        if fmt not in ("gjf", "com"):
            continue
        path = EXPORTS_DIR / f"{stem}.{fmt}"
        path.write_text(text, encoding="utf-8")
        paths[fmt] = str(path.relative_to(ROOT))
    return paths


def cancel_job(job_id: str) -> bool:
    pid_path = job_dir(job_id) / "pid"
    if not pid_path.exists():
        return False
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        os.kill(pid, signal.SIGTERM)
    except (OSError, ValueError):
        return False
    st = read_status(job_id) or {}
    st["state"] = "cancelled"
    st["error"] = "Cancelled by user"
    write_status(job_id, st)
    return True


class ViewerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt, *args):
        if str(args[0]).startswith("GET /api/"):
            return
        super().log_message(fmt, *args)

    def send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        if self.path.startswith("/api/g16"):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
        else:
            self.send_error(404)

    def read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def handle_g16_get(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/g16/env":
            cmd = current_g16_cmd()
            self.send_json(
                200,
                {
                    "available": bool(cmd),
                    "command": cmd,
                    "jobsDir": str(JOBS_DIR.relative_to(ROOT)),
                    "exportsDir": str(EXPORTS_DIR.relative_to(ROOT)),
                },
            )
            return
        if path == "/api/g16/jobs":
            self.send_json(200, {"jobs": list_jobs()})
            return
        prefix = "/api/g16/jobs/"
        if path.startswith(prefix):
            job_id = unquote(path[len(prefix):]).strip("/")
            if not job_id or "/" in job_id:
                self.send_error(404)
                return
            st = read_status(job_id)
            if not st:
                self.send_error(404)
                return
            log_path = job_dir(job_id) / f"{job_id}.log"
            st = dict(st)
            st["id"] = job_id
            st["logTail"] = tail_log(log_path)
            self.send_json(200, st)
            return
        self.send_error(404)

    def handle_g16_post(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/g16/export":
            data = self.read_json_body()
            com = (data.get("com") or "").strip()
            if not com or len(com.splitlines()) < 5:
                self.send_json(400, {"error": "Missing or invalid Gaussian input"})
                return
            label = (data.get("label") or "g16_export").strip()
            fmt = (data.get("format") or "gjf").strip().lower()
            formats = ["gjf", "com"] if fmt == "both" else [fmt]
            paths = write_gaussview_export(com, label, formats)
            if not paths:
                self.send_json(400, {"error": "format must be gjf, com, or both"})
                return
            self.send_json(200, {"paths": paths, "exportsDir": str(EXPORTS_DIR)})
            return

        if path == "/api/g16/submit":
            if not current_g16_cmd():
                self.send_json(
                    503,
                    {
                        "error": "Gaussian not found. Set G16_COMMAND or install g16 on PATH.",
                    },
                )
                return
            data = self.read_json_body()
            com = (data.get("com") or "").strip()
            if not com or len(com.splitlines()) < 5:
                self.send_json(400, {"error": "Missing or invalid .com content"})
                return
            label = (data.get("label") or "g16_job").strip()[:80]
            job_id = uuid.uuid4().hex[:12]
            jd = job_dir(job_id)
            jd.mkdir(parents=True, exist_ok=False)
            (jd / f"{job_id}.com").write_text(com + ("\n" if not com.endswith("\n") else ""), encoding="utf-8")
            route_line = ""
            for line in com.splitlines():
                if line.strip().startswith("#"):
                    route_line = line.strip()
                    break
            write_status(
                job_id,
                {
                    "state": "queued",
                    "label": label,
                    "created": utc_now(),
                    "route": route_line,
                },
            )
            thread = threading.Thread(target=run_job, args=(job_id,), daemon=True)
            thread.start()
            self.send_json(200, {"jobId": job_id, "state": "queued"})
            return

        prefix = "/api/g16/jobs/"
        if path.endswith("/cancel") and path.startswith(prefix):
            job_id = unquote(path[len(prefix): -len("/cancel")])
            if cancel_job(job_id):
                self.send_json(200, {"ok": True, "jobId": job_id})
            else:
                self.send_json(404, {"error": "Job not running"})
            return

        self.send_error(404)

    def do_GET(self):
        if self.path.startswith("/api/g16"):
            self.handle_g16_get()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/g16"):
            self.handle_g16_post()
        else:
            self.send_error(404)


def main() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    port = int(os.environ.get("VIEWER_PORT", "8765"))
    host = os.environ.get("VIEWER_HOST", "127.0.0.1")
    server = ThreadingHTTPServer((host, port), ViewerHandler)
    print(f"Serving {ROOT}", flush=True)
    print(f"Viewer:  http://{host}:{port}/viewer/", flush=True)
    if G16_CMD:
        print(f"Gaussian: {G16_CMD}", flush=True)
    else:
        print("Gaussian: NOT FOUND — export .com only until g16 is on PATH", flush=True)
    print(f"G16 jobs: {JOBS_DIR}", flush=True)
    print(f"GaussView exports: {EXPORTS_DIR}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)


if __name__ == "__main__":
    main()
