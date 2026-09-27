#!/usr/bin/env python3
"""
telemetry_ui.py - a local web interface for telemetry_overlay.py

    python telemetry_ui.py            (opens http://127.0.0.1:8765 in your browser)

Everything runs on this PC; nothing is exposed to the network. Settings are
saved as overlay_project.json next to your video files.
"""

import base64
import hashlib
import io
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import telemetry_overlay as go  # noqa: E402
go.ensure_prerequisites(need_ffmpeg=True)

from PIL import Image  # noqa: E402

PORT = 8765
PROJECT_FILE = "overlay_project.json"
CACHE_FILE = ".overlay_telemetry_cache.pkl"

DEFAULT_SETTINGS = {
    "sf_time": "", "sf_point": "", "green": "", "checkered": "", "sectors": 3,
    "min_lap": 20, "sf_width": 30, "drivers": [],
    "team": "", "car": "", "number": "", "sponsors": [], "sponsor_slots": 0, "logo": "", "logo_height": 80,
    "hide": [], "scale": 1.0, "g_max": 1.5, "g_peaks": "lap", "brake_g": 0.5, "kmh": False, "speed_max": "",
    "lap_list": 6, "tz": "",
    "height": 1080, "nvidia": True, "jobs": "", "gpu_filters": "auto", "overlay_fps": 30, "cq": 19,
    "output": "", "start": "", "duration": "",
}

LOCK = threading.Lock()
S = {
    "folder": None, "files": [], "durations": [], "raw": None, "tel": None, "laps": None,
    "settings": dict(DEFAULT_SETTINGS),
    "busy": None,                 # text while loading telemetry
    "error": None,
    "render": {"running": False, "line": "", "log": [], "done": False, "rc": None, "output": ""},
    "frame_cache": {},
}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _jsonable(o):
    """numpy scalars -> plain Python for json.dumps"""
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o).__name__}")


def list_groups(folder):
    """GoPro chapters grouped by session number: GX01NNNN, GX02NNNN ... -> NNNN"""
    groups = {}
    for name in sorted(os.listdir(folder)):
        m = re.match(r"^(G[XH])(\d\d)(\d{4})\.MP4$", name, re.I)
        if m:
            groups.setdefault(m.group(3), []).append(name)
        elif name.lower().endswith(".mp4") and not name.lower().endswith("_overlay.mp4"):
            groups.setdefault(name, []).append(name)
    out = []
    for key, names in groups.items():
        names.sort()
        size = sum(os.path.getsize(os.path.join(folder, n)) for n in names)
        out.append({"key": key, "files": names, "size_gb": round(size / 1e9, 1)})
    return out


def project_path():
    return os.path.join(S["folder"], PROJECT_FILE)


def save_project():
    if not S["folder"]:
        return
    data = {"files": S["files"], "settings": S["settings"]}
    with open(project_path(), "w") as f:
        json.dump(data, f, indent=2)


def load_project(folder):
    p = os.path.join(folder, PROJECT_FILE)
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def cache_key(paths):
    h = hashlib.md5()
    for p in paths:
        h.update(f"{os.path.basename(p)}:{os.path.getsize(p)}".encode())
    return h.hexdigest()


def load_telemetry(files):
    """Background: extract (or read cached) telemetry for the chosen files."""
    try:
        paths = [os.path.join(S["folder"], f) for f in files]
        key = cache_key(paths)
        cache = os.path.join(S["folder"], CACHE_FILE)
        raw = durations = None
        if os.path.exists(cache):
            try:
                with open(cache, "rb") as f:
                    c = pickle.load(f)
                if c.get("key") == key:
                    raw, durations = c["raw"], c["durations"]
            except Exception:
                pass
        if raw is None:
            S["busy"] = f"Reading telemetry from {len(paths)} file(s)... (a minute or so the first time)"
            raw, durations = go.extract_gopro_gps(paths, verbose=False)
            try:
                with open(cache, "wb") as f:
                    pickle.dump({"key": key, "raw": raw, "durations": durations}, f, protocol=4)
            except Exception:
                pass
        S["busy"] = "Processing GPS and accelerometer..."
        tel = go.Telemetry(raw, 0.0, verbose=False)
        with LOCK:
            S["raw"], S["durations"], S["tel"], S["files"] = raw, durations, tel, files
            S["frame_cache"].clear()
        recompute_laps()
        save_project()
    except BaseException as e:
        S["error"] = f"{e}"
        traceback.print_exc()
    finally:
        S["busy"] = None


def recompute_laps():
    tel, st = S["tel"], S["settings"]
    if tel is None:
        return
    d = S["durations"]
    kw = dict(width=float(st["sf_width"] or 30), min_lap=float(st["min_lap"] or 20),
              green=go.parse_time(st["green"] or None, d), checkered=go.parse_time(st["checkered"] or None, d),
              n_sectors=int(st["sectors"] or 3))
    try:
        if st["sf_time"]:
            laps = go.Laps.from_time(tel, go.parse_time(st["sf_time"], d), **kw)
        elif st["sf_point"]:
            lat, lon = (float(v) for v in st["sf_point"].split(","))
            laps = go.Laps.from_point(tel, lat, lon, **kw)
        else:
            laps = go.Laps.auto(tel, **kw)
        specs = [f"{x['name']}@{x['when']}" if x.get("when") else x["name"] for x in st["drivers"] if x.get("name")]
        if specs:
            try:
                laps.set_drivers(specs, d)
            except SystemExit as e:
                S["error"] = str(e)
        with LOCK:
            S["laps"] = laps
            S["frame_cache"].clear()
    except SystemExit as e:
        S["error"] = str(e)


def laps_json():
    laps, d = S["laps"], S["durations"]
    if laps is None:
        return None
    tel = laps.tel
    rows = []
    best = int(min(range(len(laps.times)), key=lambda i: laps.times[i])) if laps.times else None
    med = sorted(laps.times)[len(laps.times) // 2] if laps.times else 0
    for i, lt in enumerate(laps.times):
        pk = tel.g_peaks(laps.crossings[i], laps.crossings[i + 1])
        rows.append({
            "lap": i + 1, "driver": laps.driver_of_lap(i) or "", "time": go.fmt_lap(lt), "time_s": round(lt, 3),
            "gap": (lt - laps.times[best]) if best is not None else 0,
            "sectors": [go.fmt_sector(v) if v is not None else "" for v in laps.sector_times[i]],
            "start": laps.crossings[i], "start_txt": go.fmt_chapter(laps.crossings[i], d),
            "long": lt > 1.6 * med, "best": i == best,
            "lat_g": round(max(pk[0], pk[1]), 2), "brk_g": round(pk[3], 2),
        })
    out = {"rows": rows, "n_raw": laps.n_raw, "explain": None if laps.times else laps.explain(),
           "best": go.fmt_lap(laps.times[best]) if best is not None else None,
           "theo": None, "drivers": []}
    if laps.n_sectors >= 2 and laps.times:
        bs = laps.best_sectors(len(laps.times))
        if all(v is not None for v in bs):
            out["theo"] = go.fmt_lap(sum(bs))
            out["best_sectors"] = [go.fmt_sector(v) for v in bs]
    for name in dict.fromkeys(n for _, n in laps.drivers):
        idx = [i for i in range(len(laps.times)) if laps.driver_of_lap(i) == name]
        if idx:
            mine = [laps.times[i] for i in idx]
            out["drivers"].append({"name": name, "laps": len(mine), "best": go.fmt_lap(min(mine)),
                                   "miles": round(sum(laps.lap_dist[i] for i in idx) / 1609.344, 1)})
    return out


def settings_to_args(st, for_render=True):
    a = []
    if st["sf_time"]:
        a += ["--sf-time", st["sf_time"]]
    elif st["sf_point"]:
        a += ["--sf-point", st["sf_point"]]
    if st["green"]:
        a += ["--green", st["green"]]
    if st["checkered"]:
        a += ["--checkered", st["checkered"]]
    a += ["--sectors", str(st["sectors"]), "--min-lap", str(st["min_lap"]), "--sf-width", str(st["sf_width"])]
    for dv in st["drivers"]:
        if dv.get("name"):
            a += ["--driver", f"{dv['name']}@{dv['when']}" if dv.get("when") else dv["name"]]
    if st["team"]:
        a += ["--team", st["team"]]
    if st["car"]:
        a += ["--car", st["car"]]
    if st["number"]:
        a += ["--number", str(st["number"])]
    if st["logo"]:
        a += ["--logo", st["logo"], "--logo-height", str(st["logo_height"])]
    for sp in st["sponsors"]:
        a += ["--sponsor", sp]
    if st["sponsor_slots"]:
        a += ["--sponsor-slots", str(st["sponsor_slots"])]
    if st["hide"]:
        a += ["--hide", ",".join(st["hide"])]
    a += ["--scale", str(st["scale"]), "--g-max", str(st["g_max"]), "--g-peaks", st["g_peaks"],
          "--brake-g", str(st["brake_g"]), "--lap-list", str(st["lap_list"])]
    if st["kmh"]:
        a += ["--kmh"]
    if st["speed_max"]:
        a += ["--speed-max", str(st["speed_max"])]
    if st["tz"] != "":
        a += ["--tz", str(st["tz"])]
    if for_render:
        a += ["--height", str(st["height"]), "--overlay-fps", str(st["overlay_fps"]), "--cq", str(st["cq"]),
              "--gpu-filters", st["gpu_filters"]]
        if st["nvidia"]:
            a += ["--nvidia"]
        if st["jobs"]:
            a += ["--jobs", str(st["jobs"])]
        if st["start"]:
            a += ["--start", st["start"]]
        if st["duration"]:
            a += ["--duration", st["duration"]]
    return a


def renderer_for_preview(W, H):
    st = S["settings"]
    tz = None
    if st["tz"] != "":
        tz = go._dt.timezone(go._dt.timedelta(hours=float(st["tz"])))
    d = S["durations"]
    return go.Renderer(
        S["tel"], S["laps"], W, H, "kmh" if st["kmh"] else "mph",
        float(st["speed_max"]) if st["speed_max"] else None, go.Fonts(None), 3.0, 5.0, int(st["lap_list"]),
        set(st["hide"]), float(st["scale"]), float(st["g_max"]), tz, go.parse_time(st["green"] or None, d),
        st["g_peaks"], st["logo"] or None, list(st["sponsors"]), int(st["sponsor_slots"] or 0),
        float(st["logo_height"]), st["team"] or None, st["car"] or None, st["number"] or None, float(st["brake_g"]))


def grab_frame(t, width):
    """JPEG bytes of the source video at combined time t, `width` px wide."""
    paths = [os.path.join(S["folder"], f) for f in S["files"]]
    hw = ["-hwaccel", "cuda"] if S["settings"]["nvidia"] else []
    with tempfile.TemporaryDirectory() as td:
        inp = go.video_input_args(paths, td, t, S["durations"])
        cmd = ["ffmpeg", "-v", "error", "-y"] + hw + inp + ["-frames:v", "1", "-vf", f"scale={width}:-2",
                                                             "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "4", "-"]
        p = subprocess.run(cmd, capture_output=True, timeout=120)
        if p.returncode != 0 or not p.stdout:
            # retry without hwaccel
            cmd = ["ffmpeg", "-v", "error", "-y"] + inp + ["-frames:v", "1", "-vf", f"scale={width}:-2",
                                                            "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "4", "-"]
            p = subprocess.run(cmd, capture_output=True, timeout=120)
        if p.returncode != 0 or not p.stdout:
            raise RuntimeError(p.stderr.decode("utf8", "replace")[-400:] or "ffmpeg produced no frame")
        return p.stdout


def frame_with_overlay(t, width):
    key = ("f", round(t, 2), width)
    jpg = S["frame_cache"].get(key)
    if jpg is None:
        jpg = grab_frame(t, width)
        if len(S["frame_cache"]) > 60:
            S["frame_cache"].clear()
        S["frame_cache"][key] = jpg
    bg = Image.open(io.BytesIO(jpg)).convert("RGBA")
    if S["laps"] is not None:
        r = renderer_for_preview(bg.width, bg.height)
        bg.alpha_composite(r.frame(t))
    out = io.BytesIO()
    bg.convert("RGB").save(out, "JPEG", quality=88)
    return out.getvalue()


# ----------------------------------------------------------------------------
# rendering (subprocess of telemetry_overlay.py)
# ----------------------------------------------------------------------------

def start_render():
    R = S["render"]
    st = S["settings"]
    out = st["output"] or f"{S['files'][0].rsplit('.', 1)[0]}_overlay.mp4"
    if not os.path.isabs(out):
        out = os.path.join(S["folder"], out)
    cmd = [sys.executable, os.path.join(HERE, "telemetry_overlay.py")] + S["files"] + settings_to_args(st) + ["-o", out]
    R.update({"running": True, "line": "starting...", "log": [" ".join(cmd)], "done": False, "rc": None,
              "output": out, "cmd": cmd})
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    popen_kw = {}
    if os.name == "nt":
        popen_kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kw["start_new_session"] = True
    proc = subprocess.Popen(cmd, cwd=S["folder"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                            bufsize=0, **popen_kw)
    R["proc"] = proc

    def reader():
        buf = b""
        while True:
            ch = proc.stdout.read(1)
            if not ch:
                break
            if ch in (b"\r", b"\n"):
                line = buf.decode("utf8", "replace").rstrip()
                buf = b""
                if not line:
                    continue
                if "%" in line and "ETA" in line:
                    R["line"] = line.strip()
                else:
                    R["log"].append(line)
                    R["log"] = R["log"][-200:]
            else:
                buf += ch
        proc.wait()
        if R.get("proc") is proc:
            R["rc"] = proc.returncode if R["rc"] is None else R["rc"]
            R["running"] = False
            R["done"] = True

    threading.Thread(target=reader, daemon=True).start()


def kill_tree(proc):
    """Stop a render and everything it started (worker processes, ffmpeg)."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def cancel_render():
    R = S["render"]
    p = R.get("proc")
    if p and p.poll() is None:
        kill_tree(p)
        R["log"].append("Cancelled.")
    R["running"] = False
    R["done"] = True
    R["rc"] = R.get("rc") if R.get("rc") is not None else -1


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------

def pick_folder_dialog():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title="Choose the folder with your GoPro files")
        root.destroy()
        return path or None
    except Exception:
        return None


def pick_file_dialog():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(title="Choose an image",
                                          filetypes=[("Images", "*.png *.jpg *.jpeg *.webp"), ("All", "*.*")])
        root.destroy()
        return path or None
    except Exception:
        return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)) or body is None:
            body = json.dumps(body, default=_jsonable).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/":
                with open(os.path.join(HERE, "telemetry_ui.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if u.path == "/api/state":
                return self._send(200, self.state())
            if u.path == "/api/laps":
                return self._send(200, laps_json())
            if u.path == "/api/frame":
                t = float(q.get("t", 0))
                w = int(q.get("w", 960))
                jpg = frame_with_overlay(t, w) if q.get("overlay") == "1" else grab_frame(t, w)
                return self._send(200, jpg, "image/jpeg")
            if u.path == "/api/render/status":
                R = S["render"]
                return self._send(200, {k: R.get(k) for k in ("running", "line", "log", "done", "rc", "output")})
            if u.path == "/api/pick_folder":
                return self._send(200, {"path": pick_folder_dialog()})
            if u.path == "/api/pick_file":
                return self._send(200, {"path": pick_file_dialog()})
            return self._send(404, {"error": "not found"})
        except Exception as e:
            traceback.print_exc()
            return self._send(500, {"error": str(e)})

    def do_POST(self):
        u = urlparse(self.path)
        try:
            body = self._json()
            if u.path == "/api/open":
                folder = body.get("folder", "").strip().strip('"')
                if not os.path.isdir(folder):
                    return self._send(400, {"error": f"Not a folder: {folder}"})
                S["folder"] = folder
                S["tel"] = S["laps"] = None
                S["files"] = []
                S["error"] = None
                proj = load_project(folder)
                if proj:
                    st = dict(DEFAULT_SETTINGS)
                    st.update(proj.get("settings", {}))
                    S["settings"] = st
                    files = [f for f in proj.get("files", []) if os.path.exists(os.path.join(folder, f))]
                    if files:
                        S["files"] = files
                        threading.Thread(target=load_telemetry, args=(files,), daemon=True).start()
                else:
                    S["settings"] = dict(DEFAULT_SETTINGS)
                return self._send(200, {"groups": list_groups(folder), "state": self.state()})
            if u.path == "/api/load":
                files = body.get("files") or []
                if not files:
                    return self._send(400, {"error": "choose at least one file"})
                S["error"] = None
                threading.Thread(target=load_telemetry, args=(files,), daemon=True).start()
                return self._send(200, {"ok": True})
            if u.path == "/api/settings":
                S["settings"].update(body.get("settings", {}))
                if body.get("recompute", True):
                    S["error"] = None
                    recompute_laps()
                save_project()
                return self._send(200, {"state": self.state(), "laps": laps_json()})
            if u.path == "/api/render/start":
                if S["render"]["running"]:
                    return self._send(400, {"error": "already rendering"})
                if S["tel"] is None:
                    return self._send(400, {"error": "load a session first"})
                start_render()
                return self._send(200, {"ok": True})
            if u.path == "/api/render/cancel":
                cancel_render()
                return self._send(200, {"ok": True})
            if u.path == "/api/upload":
                # sponsor / logo image sent as base64; saved next to the videos
                name = re.sub(r"[^\w.\-]", "_", body["name"])
                dest = os.path.join(S["folder"], "overlay_images")
                os.makedirs(dest, exist_ok=True)
                path = os.path.join(dest, name)
                with open(path, "wb") as f:
                    f.write(base64.b64decode(body["data"].split(",", 1)[-1]))
                return self._send(200, {"path": path})
            return self._send(404, {"error": "not found"})
        except Exception as e:
            traceback.print_exc()
            return self._send(500, {"error": str(e)})

    def state(self):
        d = S["durations"]
        return {
            "folder": S["folder"], "files": S["files"], "durations": d, "total": sum(d) if d else 0,
            "loaded": S["tel"] is not None, "busy": S["busy"], "error": S["error"],
            "settings": S["settings"], "has_accel": bool(S["tel"] is not None and S["tel"].g_source == "accel"),
            "has_clock": bool(S["tel"] is not None and S["tel"].utc_offset is not None),
            "chapters": [{"file": f, "start": sum(d[:i]), "start_txt": go.fmt_clock(sum(d[:i]))}
                         for i, f in enumerate(S["files"])] if d else [],
            "ffmpeg": bool(__import__("shutil").which("ffmpeg")),
            "elements": go.ALL_ELEMENTS, "render": {k: S["render"].get(k) for k in ("running", "done", "rc")},
            "cmd_preview": " ".join([os.path.basename(sys.executable), "telemetry_overlay.py"] + S["files"]
                                    + settings_to_args(S["settings"]) + ["-o", S["settings"]["output"] or "..."])
            if S["files"] else "",
        }


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"Racing Telemetry Overlay running at {url}")
    print("Keep this window open. You can close and reopen the browser tab at any time;")
    print("closing THIS window stops the server and any render in progress.  (Ctrl+C to stop)")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        cancel_render()


if __name__ == "__main__":
    main()
