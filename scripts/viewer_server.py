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
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
JOBS_DIR = ROOT / "viewer" / "g16_jobs"
EXPORTS_DIR = ROOT / "viewer" / "gaussview_exports"
WORK_ROOT = Path("/tmp/dna_caox_g16")
LOG_TAIL_BYTES = 12000
GEOM_PDBS = {
    "sphere": ROOT / "DNA_CaOx_growth_whewellite30A_relaxed.pdb",
    "slab": ROOT / "DNA_CaOx_growth_whewellite30A_dls.pdb",
    "allp": ROOT / "DNA_CaOx_growth_whewellite30A_allP.pdb",
    "local": ROOT / "DNA_CaOx_growth_whewellite20A_allP_dls.pdb",
    "local10": ROOT / "DNA_CaOx_growth_whewellite10A_allP_dls.pdb",
    "altp": ROOT / "DNA_CaOx_growth_whewellite30A_altP_omm.pdb",
    "gel": ROOT / "DNA_CaOx_gel_first_omm.pdb",
    "shell15": ROOT / "DNA_CaOx_gel_first_shell15A_omm.pdb",
    "gel_altp_geom": ROOT / "DNA_CaOx_gel_altP_geom_omm.pdb",
    "shell_lattice": ROOT / "DNA_CaOx_gel_altP_geom_shell_lattice_omm.pdb",
    "shell_lattice_seeded": ROOT / "DNA_CaOx_gel_altP_geom_shell_lattice_seeded_omm.pdb",
    "templating_gel": ROOT / "DNA_CaOx_templating_gel_omm.pdb",
    "templating_gel_thick": ROOT / "DNA_CaOx_templating_gel_thick_omm.pdb",
    "templating_gel_10shell": ROOT / "DNA_CaOx_templating_gel_10shell_omm.pdb",
    "templating_gel_15shell": ROOT / "DNA_CaOx_templating_gel_15shell_omm.pdb",
    "templating_nodna": ROOT / "DNA_CaOx_templating_gel_nodna_omm.pdb",
}
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


def normalize_gaussian_com(text: str) -> str:
    """Strip wrapping whitespace, then end with a blank line (required by l101)."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""
    return text + "\n\n"


ATOMIC_Z = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Na": 11,
    "Mg": 12,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "K": 19,
    "Ca": 20,
}


def fit_charge_to_multiplicity(charge: int, ztot: int, mult: int) -> int:
    """Singlet/triplet need even electron counts; doublets need odd."""
    want_odd = (int(mult) - 1) % 2 == 1
    nelec_odd = ((int(ztot) - int(charge)) % 2) == 1
    if nelec_odd == want_odd:
        return int(charge)
    return int(charge) - 1 if int(charge) <= 0 else int(charge) + 1


def ensure_com_electron_parity(com: str) -> tuple[str, int | None, int | None]:
    """Nudge charge by 1 if multiplicity and electron count are incompatible.

    Returns (com, old_charge, new_charge). new_charge is None if unchanged.
    """
    lines = com.splitlines()
    atom_re = re.compile(r"^([A-Za-z]{1,2})\s+")
    first_atom = None
    ztot = 0
    for i, line in enumerate(lines):
        s = line.strip()
        m = atom_re.match(s)
        if not m:
            continue
        parts = s.split()
        if len(parts) < 4:
            continue
        try:
            float(parts[1])
            float(parts[2])
            float(parts[3])
        except ValueError:
            continue
        el_raw = m.group(1)
        el = el_raw[0].upper() + (el_raw[1:].lower() if len(el_raw) > 1 else "")
        z = ATOMIC_Z.get(el)
        if z is None:
            continue
        if first_atom is None:
            first_atom = i
        ztot += z
    if first_atom is None:
        return com, None, None
    charge_idx = None
    charge = mult = None
    for j in range(first_atom - 1, -1, -1):
        parts = lines[j].split()
        if len(parts) != 2:
            continue
        try:
            charge, mult = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        charge_idx = j
        break
    if charge_idx is None or charge is None or mult is None:
        return com, None, None
    new_q = fit_charge_to_multiplicity(charge, ztot, mult)
    if new_q == charge:
        return com, charge, None
    lines[charge_idx] = f"{new_q} {mult}"
    return normalize_gaussian_com("\n".join(lines)), charge, new_q


def gaussian_env(g16_cmd: str, scratch: Path) -> dict[str, str]:
    env = os.environ.copy()
    g16_dir = Path(g16_cmd).resolve().parent
    g16root = g16_dir.parent
    env["g16root"] = str(g16root)
    env["GAUSS_EXEDIR"] = f"{g16_dir / 'bsd'}:{g16_dir}"
    env["GAUSS_LEXEDIR"] = str(g16_dir / "linda-exe")
    env["GAUSS_ARCHDIR"] = str(g16_dir / "arch")
    env["GAUSS_BSDDIR"] = str(g16_dir / "bsd")
    env["G16BASIS"] = str(g16_dir / "basis")
    env["GAUSS_SCRDIR"] = str(scratch)
    env["PATH"] = f"{g16_dir / 'bsd'}:{g16_dir}:{env.get('PATH', '')}"
    return env


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
                fh.seek(-nbytes, os.SEEK_END)
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

    # iCloud/repo paths contain spaces; Gaussian l101 often dies with
    # "End of file in ZSymb" if scratch or cwd has spaces.
    work = WORK_ROOT / job_id
    scratch = work / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    work_com = work / f"{job_id}.com"
    shutil.copy2(com_path, work_com)
    env = gaussian_env(g16_cmd, scratch)

    try:
        with work_com.open("rb") as inf, log_path.open("wb", buffering=0) as logfh:
            proc = subprocess.Popen(
                [g16_cmd],
                cwd=str(work),
                stdin=inf,
                stdout=logfh,
                stderr=subprocess.STDOUT,
                env=env,
            )
            pid_path.write_text(str(proc.pid), encoding="utf-8")
            exit_code = proc.wait()
        pid_path.unlink(missing_ok=True)
        for extra in work.glob("*.chk"):
            shutil.copy2(extra, jd / extra.name)
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
    text = normalize_gaussian_com(com)
    paths: dict[str, str] = {}
    for fmt in formats:
        if fmt not in ("gjf", "com"):
            continue
        path = EXPORTS_DIR / f"{stem}.{fmt}"
        path.write_text(text, encoding="utf-8")
        paths[fmt] = str(path.relative_to(ROOT))
    return paths


def pdb_element(line: str) -> str:
    el = line[76:78].strip() if len(line) >= 78 else ""
    if not el:
        name = line[12:16].strip()
        el = "Ca" if name.upper().startswith("CA") else name[:1]
    if el.lower() == "ca":
        return "Ca"
    if len(el) == 1:
        return el.upper()
    return el[0].upper() + el[1:].lower()


def read_pdb_export_atoms(
    pdb_path: Path,
    *,
    include_dna: bool,
    include_mineral: bool,
    include_water: bool,
) -> tuple[list[tuple[str, float, float, float]], list[str]]:
    """Return Cartesian atoms and the original PDB records kept."""
    atoms: list[tuple[str, float, float, float]] = []
    records: list[str] = []
    for line in pdb_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        res = line[17:20].strip()
        if res == "NUC" and not include_dna:
            continue
        if res == "WHW" and not include_mineral:
            continue
        if res in ("HOH", "WAT", "SOL") and not include_water:
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        atoms.append((pdb_element(line), x, y, z))
        records.append(line[:80].rstrip())
    return atoms, records


def estimate_pdb_charge(pdb_path: Path, records: list[str]) -> int:
    n_p = n_ca = n_ox_c = 0
    for line in records:
        el = pdb_element(line)
        res = line[17:20].strip()
        if el == "P":
            n_p += 1
        elif el == "Ca":
            n_ca += 1
        elif el == "C" and res == "WHW":
            n_ox_c += 1
    return n_p * -1 + n_ca * 2 + (n_ox_c // 2) * -2


def write_gaussview_from_pdb(
    geometry: str,
    *,
    label: str,
    route: str,
    mem: str,
    nproc: int,
    charge: int | None,
    mult: int,
    include_dna: bool,
    include_mineral: bool,
    include_water: bool,
    formats: list[str],
) -> dict:
    pdb_path = GEOM_PDBS.get(geometry)
    if not pdb_path or not pdb_path.exists():
        raise FileNotFoundError(f"No source PDB for geometry {geometry!r}")
    atoms, records = read_pdb_export_atoms(
        pdb_path,
        include_dna=include_dna,
        include_mineral=include_mineral,
        include_water=include_water,
    )
    if len(atoms) < 3:
        raise ValueError("PDB export produced fewer than 3 atoms")
    q = estimate_pdb_charge(pdb_path, records) if charge is None else int(charge)
    ztot = sum(ATOMIC_Z.get(el, 0) for el, _, _, _ in atoms)
    q = fit_charge_to_multiplicity(q, ztot, max(1, int(mult)))
    title = f"{geometry} full model ({len(atoms)} atoms) from {pdb_path.name}"
    lines = [
        f"%chk={safe_export_stem(label)}_g16.chk",
        f"%mem={mem}",
        f"%nprocshared={max(1, int(nproc))}",
        route.strip() or "#p B3LYP/6-31G(d)",
        "",
        title,
        "",
        f"{q} {max(1, int(mult))}",
    ]
    for el, x, y, z in atoms:
        lines.append(f"{el:<2s} {x:12.6f} {y:12.6f} {z:12.6f}")
    com = normalize_gaussian_com("\n".join(lines))
    paths = write_gaussview_export(com, label, formats)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pdb_out = EXPORTS_DIR / f"{safe_export_stem(label)}_g16.pdb"
    pdb_body = "\n".join(records) + "\nEND\n"
    pdb_out.write_text(pdb_body, encoding="utf-8")
    paths["pdb"] = str(pdb_out.relative_to(ROOT))
    paths["sourcePdb"] = str(pdb_path.relative_to(ROOT))
    return {"paths": paths, "nAtoms": len(atoms), "charge": q, "geometry": geometry}


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
            parsed = urlparse(self.path)
            job_id = unquote(parsed.path[len(prefix):]).strip("/")
            if not job_id or "/" in job_id:
                self.send_error(404)
                return
            st = read_status(job_id)
            if not st:
                self.send_error(404)
                return
            log_path = job_dir(job_id) / f"{job_id}.log"
            tail_n = LOG_TAIL_BYTES
            if parsed.query:
                raw = (parse_qs(parsed.query).get("tail") or [""])[0]
                try:
                    tail_n = max(2000, min(int(raw), 500_000))
                except ValueError:
                    tail_n = LOG_TAIL_BYTES
            st = dict(st)
            st["id"] = job_id
            st["logBytes"] = log_path.stat().st_size if log_path.exists() else 0
            st["logTail"] = tail_log(log_path, tail_n)
            self.send_json(200, st)
            return
        self.send_error(404)

    def handle_g16_post(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/g16/export":
            data = self.read_json_body()
            source = (data.get("source") or "com").strip().lower()
            label = (data.get("label") or "g16_export").strip()
            fmt = (data.get("format") or "gjf").strip().lower()
            formats = ["gjf", "com"] if fmt == "both" else [fmt]
            if source == "pdb":
                geom = (data.get("geometry") or "").strip()
                if geom not in GEOM_PDBS:
                    self.send_json(400, {"error": f"Unknown geometry {geom!r}"})
                    return
                try:
                    payload = write_gaussview_from_pdb(
                        geom,
                        label=label,
                        route=(data.get("route") or "").strip(),
                        mem=(data.get("mem") or "128GB").strip(),
                        nproc=int(data.get("nproc") or 28),
                        charge=data.get("charge"),
                        mult=int(data.get("mult") or 1),
                        include_dna=bool(data.get("includeDna", True)),
                        include_mineral=bool(data.get("includeOxalate", True)),
                        include_water=bool(data.get("includeWater", True)),
                        formats=formats,
                    )
                except (FileNotFoundError, ValueError) as exc:
                    self.send_json(400, {"error": str(exc)})
                    return
                payload["exportsDir"] = str(EXPORTS_DIR)
                self.send_json(200, payload)
                return
            com = normalize_gaussian_com(data.get("com") or "")
            if not com or len(com.splitlines()) < 5:
                self.send_json(400, {"error": "Missing or invalid Gaussian input"})
                return
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
            com = normalize_gaussian_com(data.get("com") or "")
            if not com or len(com.splitlines()) < 5:
                self.send_json(400, {"error": "Missing or invalid .com content"})
                return
            com, old_q, new_q = ensure_com_electron_parity(com)
            label = (data.get("label") or "g16_job").strip()[:80]
            job_id = uuid.uuid4().hex[:12]
            jd = job_dir(job_id)
            jd.mkdir(parents=True, exist_ok=False)
            (jd / f"{job_id}.com").write_text(com, encoding="utf-8")
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
            payload = {"jobId": job_id, "state": "queued"}
            if new_q is not None:
                payload["charge"] = new_q
                payload["chargeAdjustedFrom"] = old_q
            self.send_json(200, payload)
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
