#!/usr/bin/env python3
"""
telemetry_overlay.py - Racing telemetry overlay for GoPro footage.

Reads the GPS and accelerometer data that GoPro cameras (Hero 5-11) embed in
their MP4 files, detects laps from a start/finish line, and renders a new video
with a lap timer, sector splits, speedometer, G-meter, track map and more.

Requirements: Python 3.8+, numpy, Pillow, and ffmpeg/ffprobe on your PATH.
    pip install numpy pillow

Typical use:
    # 1. Check laps quickly (no video rendering). Writes a map PNG + lap table.
    python telemetry_overlay.py GX010123.MP4 --laps-only

    # 2. Preview one still frame to check the layout
    python telemetry_overlay.py GX010123.MP4 --frame 2:00

    # 3. Render the full video
    python telemetry_overlay.py GX010123.MP4 GX020123.MP4 -o session.mp4 --height 1080 --nvidia

Run with --help for all options.
"""

import argparse
import bisect
import csv
import datetime as _dt
import math
import multiprocessing as mp
import os
import queue
import threading
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time



# ----------------------------------------------------------------------------
# Prerequisites: offer to install what's missing instead of crashing
# ----------------------------------------------------------------------------

def _ask(question):
    if not sys.stdin or not sys.stdin.isatty():
        return True
    try:
        return input(question + " [Y/n] ").strip().lower() in ("", "y", "yes")
    except EOFError:
        return True


def _find_winget_ffmpeg():
    """After a winget install the new PATH only applies to new windows; look
    for the ffmpeg it just put down so this run can carry on."""
    import glob
    base = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages")
    for exe in glob.glob(os.path.join(base, "Gyan.FFmpeg*", "**", "ffmpeg.exe"), recursive=True):
        return os.path.dirname(exe)
    return None


def ensure_prerequisites(need_ffmpeg=True):
    """Check numpy, Pillow and FFmpeg; offer to install anything missing."""
    missing = []
    for mod, pkg in (("numpy", "numpy"), ("PIL", "pillow")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing Python packages: {', '.join(missing)}")
        if _ask("Install them now with pip?"):
            subprocess.call([sys.executable, "-m", "pip", "install"] + missing)
            for mod in ("numpy", "PIL"):
                try:
                    __import__(mod)
                except ImportError:
                    raise SystemExit(f"Still can't import {mod}. Try:  {sys.executable} -m pip install {' '.join(missing)}")
        else:
            raise SystemExit(f"Install them with:  {sys.executable} -m pip install {' '.join(missing)}")
    if need_ffmpeg and (shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None):
        if os.name == "nt":
            found = _find_winget_ffmpeg()
            if found:
                os.environ["PATH"] = found + os.pathsep + os.environ.get("PATH", "")
            else:
                print("FFmpeg was not found on this PC.")
                if shutil.which("winget") and _ask("Install FFmpeg now with winget (Gyan.FFmpeg, the full build)?"):
                    subprocess.call(["winget", "install", "-e", "--id", "Gyan.FFmpeg", "--accept-package-agreements",
                                     "--accept-source-agreements"])
                    found = _find_winget_ffmpeg()
                    if found:
                        os.environ["PATH"] = found + os.pathsep + os.environ.get("PATH", "")
                if shutil.which("ffmpeg") is None:
                    raise SystemExit("FFmpeg is still not available. Install it (winget install Gyan.FFmpeg, or "
                                     "download from gyan.dev and add its bin folder to PATH), then open a new "
                                     "window and try again.")
        elif sys.platform == "darwin":
            raise SystemExit("FFmpeg not found. Install it with:  brew install ffmpeg")
        else:
            raise SystemExit("FFmpeg not found. Install it with your package manager, e.g.  sudo apt install ffmpeg")


ensure_prerequisites(need_ffmpeg=False)
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

MS_TO_MPH = 2.2369363
MS_TO_KMH = 3.6
EARTH_R = 6371008.8
G0 = 9.80665

# ----------------------------------------------------------------------------
# MP4 parsing (just enough to find the GoPro 'gpmd' telemetry track)
# ----------------------------------------------------------------------------

CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf", b"udta"}


def _iter_boxes(f, start, end):
    pos = start
    while pos + 8 <= end:
        f.seek(pos)
        hdr = f.read(8)
        if len(hdr) < 8:
            return
        size, typ = struct.unpack(">I4s", hdr)
        hlen = 8
        if size == 1:
            size = struct.unpack(">Q", f.read(8))[0]
            hlen = 16
        elif size == 0:
            size = end - pos
        if size < hlen:
            return
        yield typ, pos + hlen, pos + size
        pos += size


def _read(f, start, end):
    f.seek(start)
    return f.read(end - start)


def mp4_info(path):
    """Return (movie_duration_s, [telemetry tracks]) where each telemetry track
    is a list of (file_offset, size, start_s, duration_s) samples."""
    tracks = []
    duration = 0.0
    fsize = os.path.getsize(path)
    with open(path, "rb") as f:
        moov = None
        for typ, s, e in _iter_boxes(f, 0, fsize):
            if typ == b"moov":
                moov = (s, e)
        if moov is None:
            raise ValueError(f"{path}: no 'moov' box - not a valid MP4?")
        for typ, s, e in _iter_boxes(f, *moov):
            if typ == b"mvhd":
                d = _read(f, s, e)
                if d[0] == 1:
                    ts, dur = struct.unpack(">IQ", d[20:32])
                else:
                    ts, dur = struct.unpack(">II", d[12:20])
                duration = dur / ts if ts else 0.0
            elif typ == b"trak":
                t = _parse_trak(f, s, e)
                if t is not None:
                    tracks.append(t)
    return duration, tracks


def _parse_trak(f, s, e):
    info = {}

    def walk(s, e):
        for typ, cs, ce in _iter_boxes(f, s, e):
            if typ in CONTAINERS:
                walk(cs, ce)
            elif typ in (b"mdhd", b"hdlr", b"stsd", b"stts", b"stsz", b"stsc", b"stco", b"co64"):
                info[typ] = _read(f, cs, ce)

    walk(s, e)
    stsd = info.get(b"stsd")
    if not stsd or len(stsd) < 16 or stsd[12:16] != b"gpmd":
        return None

    mdhd = info[b"mdhd"]
    timescale = struct.unpack(">I", mdhd[20:24] if mdhd[0] == 1 else mdhd[12:16])[0]

    # sample durations
    d = info[b"stts"]
    n = struct.unpack(">I", d[4:8])[0]
    durs = []
    for i in range(n):
        cnt, delta = struct.unpack(">II", d[8 + 8 * i: 16 + 8 * i])
        durs.extend([delta] * cnt)

    # sample sizes
    d = info[b"stsz"]
    fixed, count = struct.unpack(">II", d[4:12])
    sizes = [fixed] * count if fixed else list(struct.unpack(f">{count}I", d[12:12 + 4 * count]))

    # chunk offsets
    if b"co64" in info:
        d = info[b"co64"]
        n = struct.unpack(">I", d[4:8])[0]
        chunk_offs = list(struct.unpack(f">{n}Q", d[8:8 + 8 * n]))
    else:
        d = info[b"stco"]
        n = struct.unpack(">I", d[4:8])[0]
        chunk_offs = list(struct.unpack(f">{n}I", d[8:8 + 4 * n]))

    # sample-to-chunk
    d = info[b"stsc"]
    n = struct.unpack(">I", d[4:8])[0]
    stsc = [struct.unpack(">III", d[8 + 12 * i: 20 + 12 * i]) for i in range(n)]

    offsets = []
    si = 0
    for ri, (first, per_chunk, _) in enumerate(stsc):
        last = stsc[ri + 1][0] - 1 if ri + 1 < len(stsc) else len(chunk_offs)
        for chunk in range(first, last + 1):
            off = chunk_offs[chunk - 1]
            for _ in range(per_chunk):
                if si >= len(sizes):
                    break
                offsets.append(off)
                off += sizes[si]
                si += 1

    samples = []
    t = 0
    for i, off in enumerate(offsets):
        dur = durs[i] if i < len(durs) else (durs[-1] if durs else timescale)
        samples.append((off, sizes[i], t / timescale, dur / timescale))
        t += dur
    return samples


# ----------------------------------------------------------------------------
# GPMF parsing (GoPro Metadata Format) - GPS5 (Hero 5-10) and GPS9 (Hero 11)
# ----------------------------------------------------------------------------

_GPMF_FMT = {
    "b": "b", "B": "B", "s": "h", "S": "H", "l": "i", "L": "I",
    "f": "f", "d": "d", "j": "q", "J": "Q", "q": "i", "Q": "q", "F": "4s",
}


def _klv(buf, start, end):
    pos = start
    while pos + 8 <= end:
        key = buf[pos:pos + 4]
        typ = buf[pos + 4]
        ssize = buf[pos + 5]
        rep = struct.unpack(">H", buf[pos + 6:pos + 8])[0]
        ds = pos + 8
        dlen = ssize * rep
        pos = ds + ((dlen + 3) & ~3)
        if key == b"\0\0\0\0":
            continue
        yield key, typ, ssize, rep, ds


def _numeric(buf, typ, ssize, rep, ds):
    c = _GPMF_FMT.get(chr(typ))
    if c is None:
        return []
    sz = struct.calcsize(">" + c)
    per = max(1, ssize // sz)
    vals = struct.unpack_from(">" + c * (per * rep), buf, ds)
    return [vals[i * per:(i + 1) * per] for i in range(rep)]


def _expand_type(s):
    out = []
    i = 0
    while i < len(s):
        ch = s[i]
        i += 1
        if i < len(s) and s[i] == "[":
            j = s.index("]", i)
            out.extend([ch] * int(s[i + 1:j]))
            i = j + 1
        else:
            out.append(ch)
    return out




def _parse_strm(buf, s, e, out):
    scal = None
    typestr = None
    fix = None
    dop = None
    for key, typ, ssize, rep, ds in _klv(buf, s, e):
        if typ == 0:
            continue
        if key == b"SCAL":
            scal = [v for row in _numeric(buf, typ, ssize, rep, ds) for v in row]
        elif key == b"TYPE":
            typestr = buf[ds:ds + ssize * rep].split(b"\0")[0].decode("latin1")
        elif key == b"GPSF":
            fix = _numeric(buf, typ, ssize, rep, ds)[0][0]
        elif key == b"GPSP":
            dop = _numeric(buf, typ, ssize, rep, ds)[0][0] / 100.0
        elif key == b"GPSU":
            out["GPSU"] = buf[ds:ds + ssize * rep].split(b"\0")[0].decode("latin1")
        elif key == b"GPS5":
            rows = _numeric(buf, typ, ssize, rep, ds)
            sc = scal or [1]
            for r in rows:
                v = [r[k] / (sc[k] if len(sc) > 1 else sc[0]) for k in range(min(5, len(r)))]
                # lat, lon, alt, speed2d, fix, dop
                out["GPS5"].append((v[0], v[1], v[2], v[3],
                                    fix if fix is not None else 3,
                                    dop if dop is not None else 1.0))
        elif key == b"GPS9":
            if chr(typ) == "?" and typestr:
                fmt = ">" + "".join(_GPMF_FMT[c] for c in _expand_type(typestr))
            else:
                fmt = ">" + "i" * 7 + "HH"
            sz = struct.calcsize(fmt)
            sc = scal or [1] * 9
            for i in range(rep):
                r = struct.unpack_from(fmt, buf, ds + i * ssize) if sz <= ssize else None
                if r is None:
                    continue
                v = [r[k] / (sc[k] if len(sc) > 1 else sc[0]) for k in range(len(r))]
                out["GPS9"].append((v[0], v[1], v[2], v[3],
                                    int(v[8]) if len(v) > 8 else 3,
                                    v[7] if len(v) > 7 else 1.0))
        elif key == b"ACCL" and chr(typ) in ("s", "l", "f"):
            dt = {"s": ">i2", "l": ">i4", "f": ">f4"}[chr(typ)]
            ncol = ssize // np.dtype(dt).itemsize
            if ncol >= 3:
                a = np.frombuffer(buf, dtype=dt, count=ncol * rep, offset=ds).reshape(rep, ncol)[:, :3]
                sc = np.array(scal[:3] if scal and len(scal) >= 3 else [scal[0] if scal else 1] * 3, dtype=float)
                out["ACCL"] = a.astype(float) / sc


def _parse_payload(buf):
    out = {"GPS5": [], "GPS9": [], "ACCL": None, "GPSU": None}

    def walk(s, e):
        for key, typ, ssize, rep, ds in _klv(buf, s, e):
            if typ == 0:
                if key == b"STRM":
                    _parse_strm(buf, ds, ds + ssize * rep, out)
                else:
                    walk(ds, ds + ssize * rep)

    walk(0, len(buf))
    return out


def _gpsu_to_posix(s):
    try:
        base = _dt.datetime.strptime(s[:12], "%y%m%d%H%M%S").replace(tzinfo=_dt.timezone.utc)
        frac = float("0" + s[12:]) if len(s) > 12 else 0.0
        return base.timestamp() + frac
    except ValueError:
        return None


def extract_gopro_gps(paths, verbose=True):
    """Extract GPS (+ accelerometer, altitude, clock) from one or more chaptered
    GoPro MP4s. Returns (dict of numpy arrays, list of per-file durations)."""
    all_rows = {"GPS5": [], "GPS9": []}
    acc_t, acc_v = [], []
    clock = []
    durations = []
    offset = 0.0
    for p in paths:
        dur, tracks = mp4_info(p)
        durations.append(dur)
        if not tracks:
            print(f"WARNING: {os.path.basename(p)} has no GoPro telemetry track", file=sys.stderr)
        with open(p, "rb") as f:
            for samples in tracks[:1]:
                for off, size, st, sd in samples:
                    f.seek(off)
                    res = _parse_payload(f.read(size))
                    for k in ("GPS5", "GPS9"):
                        rows = res[k]
                        n = len(rows)
                        for i, r in enumerate(rows):
                            all_rows[k].append((offset + st + sd * i / n,) + r)
                    if res["ACCL"] is not None and len(res["ACCL"]):
                        n = len(res["ACCL"])
                        acc_t.append(offset + st + sd * np.arange(n) / n)
                        acc_v.append(res["ACCL"])
                    if res["GPSU"] and res["GPS5"]:
                        ts = _gpsu_to_posix(res["GPSU"])
                        if ts and res["GPS5"][0][4] >= 2:
                            clock.append((offset + st, ts))
        offset += dur

    # Prefer whichever stream has more good samples (Hero 11 writes both)
    def good(rows):
        return [r for r in rows if r[5] >= 2 and r[6] < 10.0 and not (r[1] == 0 and r[2] == 0)]

    g5, g9 = good(all_rows["GPS5"]), good(all_rows["GPS9"])
    src, rows = ("GPS9", g9) if len(g9) > len(g5) else ("GPS5", g5)
    total = len(all_rows[src])
    if verbose:
        print(f"Telemetry: {src}, {len(rows)} usable GPS samples "
              f"({total - len(rows)} dropped for poor fix) over {sum(durations):.1f}s of video")
    if len(rows) < 10:
        raise SystemExit(
            "Not enough GPS data with a good fix. Make sure GPS is turned ON in the camera "
            "settings and the camera had a lock (clear sky view) while recording.")
    a = np.array(rows, dtype=float)
    out = {"t": a[:, 0], "lat": a[:, 1], "lon": a[:, 2], "alt": a[:, 3], "speed": a[:, 4],
           "acc_t": None, "acc": None, "utc_offset": None}
    if acc_v:
        out["acc_t"] = np.concatenate(acc_t)
        out["acc"] = np.concatenate(acc_v)
        if verbose:
            rate = len(out["acc_t"]) / max(sum(durations), 1)
            print(f"Accelerometer: {len(out['acc_t'])} samples ({rate:.0f} Hz)")
    if len(clock) >= 3:
        diffs = np.array([ts - vt for vt, ts in clock])
        out["utc_offset"] = float(np.median(diffs))
    return out, durations


def load_csv_telemetry(path):
    """CSV with header containing t (seconds from video start), lat, lon and
    optionally speed (m/s) and alt (m). Handy for external loggers or testing."""
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        cols = {c.lower().strip(): c for c in rd.fieldnames}
        rows = list(rd)

    def col(*names):
        for n in names:
            if n in cols:
                return np.array([float(r[cols[n]] or "nan") for r in rows])
        return None

    t, lat, lon = col("t", "time", "seconds"), col("lat", "latitude"), col("lon", "lng", "longitude")
    if t is None or lat is None or lon is None:
        raise SystemExit("CSV needs columns: t, lat, lon (and optionally speed in m/s, alt in m)")
    return {"t": t, "lat": lat, "lon": lon, "speed": col("speed", "speed_ms"), "alt": col("alt", "altitude", "ele"),
            "acc_t": None, "acc": None, "utc_offset": None}


def smooth(v, k):
    """Centered moving average with window 2k+1 (edge padded)."""
    if k <= 0 or len(v) < 2 * k + 1:
        return np.asarray(v, dtype=float)
    pad = np.pad(np.asarray(v, dtype=float), k, mode="edge")
    return np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), mode="same")[k:-k]


# ----------------------------------------------------------------------------
# Telemetry processing
# ----------------------------------------------------------------------------

class Telemetry:
    def __init__(self, d, offset=0.0, verbose=True):
        order = np.argsort(d["t"])
        t = d["t"][order] + offset
        lat, lon = d["lat"][order], d["lon"][order]
        self.lat0, self.lon0 = float(np.median(lat)), float(np.median(lon))
        x, y = self.to_xy(lat, lon)
        sp_raw = d["speed"][order] if d.get("speed") is not None else None
        alt_raw = d["alt"][order] if d.get("alt") is not None else None

        # 1) drop points nowhere near the session (a camera's first fixes can be
        #    hundreds of km off) - the median position is a robust "where we were"
        far = np.hypot(x, y) > 30_000
        if far.any() and (~far).sum() >= 10:
            if verbose:
                print(f"Dropped {int(far.sum())} GPS point(s) more than 30 km from the session (bad fixes)")
            t, x, y = t[~far], x[~far], y[~far]
            if sp_raw is not None:
                sp_raw = sp_raw[~far]
            if alt_raw is not None:
                alt_raw = alt_raw[~far]

        # 2) reject wild jumps between consecutive points; if the anchor itself
        #    turns out to be the bad point (everything after it looks like a jump)
        #    re-anchor rather than throwing the rest of the session away
        keep = [0]
        rejected = 0
        for i in range(1, len(t)):
            j = keep[-1]
            dt = t[i] - t[j]
            if dt <= 0:
                continue
            if math.hypot(x[i] - x[j], y[i] - y[j]) / dt < 120:
                keep.append(i)
                rejected = 0
            else:
                rejected += 1
                if rejected >= 10:
                    keep.pop()
                    keep.append(i)
                    rejected = 0
        keep = np.array(keep)
        if len(keep) < 10:
            raise SystemExit("Almost all GPS points were rejected as jumps - the GPS data looks corrupt. "
                             "Try --dump-telemetry to inspect it.")
        self.t, self.x, self.y = t[keep], x[keep], y[keep]

        if sp_raw is not None and not np.all(np.isnan(sp_raw)):
            sp = np.nan_to_num(sp_raw[keep])
        else:
            sp = np.hypot(np.gradient(self.x), np.gradient(self.y)) / np.maximum(np.gradient(self.t), 1e-3)
        self.speed = smooth(sp, 3)                     # light smoothing for the speedo
        self.speed_max_sofar = np.maximum.accumulate(self.speed)

        self.alt = None
        if alt_raw is not None and not np.all(np.isnan(alt_raw)):
            self.alt = smooth(np.nan_to_num(alt_raw[keep]), 15)   # ~3 s at 10 Hz

        seg = np.hypot(np.diff(self.x), np.diff(self.y))
        self.dist = np.concatenate([[0.0], np.cumsum(seg)])

        # yaw rate from the GPS track (for G-meter alignment / fallback)
        dt = np.maximum(np.gradient(self.t), 1e-3)
        hdg = np.unwrap(np.arctan2(np.gradient(self.y), np.gradient(self.x)))
        self.omega = smooth(np.gradient(hdg) / dt, 3)
        self.a_lon_gps = smooth(np.gradient(self.speed) / dt, 3)
        self.a_lat_gps = -self.speed * self.omega        # right turn = positive

        self.utc_offset = d.get("utc_offset")
        self.g_t = self.g_lon = self.g_lat = None
        self.g_source = None
        if d.get("acc") is not None:
            self._align_accel(d["acc_t"], d["acc"], verbose)
        if self.g_t is None:
            # fall back to GPS-derived g (10 Hz, smoother but usable)
            self.g_t = self.t
            self.g_lon = smooth(self.a_lon_gps, 2) / G0
            self.g_lat = smooth(self.a_lat_gps, 2) / G0
            self.g_source = "gps"

    def _align_accel(self, at, acc, verbose):
        """Work out which way the camera is pointing from the data itself:
        gravity gives 'up', correlation with GPS acceleration gives 'forward'."""
        order = np.argsort(at)
        at, acc = at[order], acc[order]
        if len(at) < 1000:
            return
        # smooth ~0.2 s and decimate to ~50 Hz
        rate = len(at) / max(at[-1] - at[0], 1e-3)
        k = max(1, int(0.1 * rate))
        acc = np.column_stack([smooth(acc[:, i], k) for i in range(3)])
        step = max(1, int(round(rate / 50)))
        at, acc = at[::step], acc[::step]

        up = np.median(acc, axis=0)
        n = np.linalg.norm(up)
        if n < 5 or n > 15:
            if verbose:
                print(f"WARNING: accelerometer gravity magnitude {n:.1f} m/s^2 looks wrong; using GPS for G-meter")
            return
        up /= n
        h = acc - np.outer(acc @ up, up)                 # horizontal specific force (camera frame)

        v = np.interp(at, self.t, self.speed)
        a_lon = np.interp(at, self.t, self.a_lon_gps)
        a_lat = np.interp(at, self.t, self.a_lat_gps)
        in_gps = (at >= self.t[0]) & (at <= self.t[-1])
        moving = (v > 8) & in_gps
        if moving.sum() < 500:
            return
        e1 = np.cross(up, [1.0, 0, 0])
        if np.linalg.norm(e1) < 0.1:
            e1 = np.cross(up, [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(up, e1)
        A = np.column_stack([h @ e1, h @ e2])[moving]
        beta, *_ = np.linalg.lstsq(A, a_lon[moving], rcond=None)
        f = beta[0] * e1 + beta[1] * e2
        if np.linalg.norm(f) < 1e-6:
            return
        f /= np.linalg.norm(f)
        r = np.cross(f, up)                              # right-hand side of the car
        g_lon, g_lat = h @ f, h @ r
        c_lon = np.corrcoef(g_lon[moving], a_lon[moving])[0, 1]
        c_lat = np.corrcoef(g_lat[moving], a_lat[moving])[0, 1]
        if c_lat < 0:
            r, g_lat, c_lat = -r, -g_lat, -c_lat
        if verbose:
            print(f"Accelerometer aligned to the car (agreement with GPS: forward {c_lon:.2f}, lateral {c_lat:.2f})")
        if c_lon < 0.4 or c_lat < 0.4:
            if verbose:
                print("  Agreement is poor - the G-meter will use GPS-derived acceleration instead")
            return
        self.g_t, self.g_lon, self.g_lat = at, g_lon / G0, g_lat / G0
        self.g_source = "accel"

    def to_xy(self, lat, lon):
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        x = np.radians(lon - self.lon0) * EARTH_R * math.cos(math.radians(self.lat0))
        y = np.radians(lat - self.lat0) * EARTH_R
        return x, y

    def at(self, T):
        return (float(np.interp(T, self.t, self.x)), float(np.interp(T, self.t, self.y)),
                float(np.interp(T, self.t, self.speed)), float(np.interp(T, self.t, self.dist)))

    def g_at(self, T):
        return float(np.interp(T, self.g_t, self.g_lon)), float(np.interp(T, self.g_t, self.g_lat))

    def g_peaks(self, t0, t1):
        """(left, right, accel, brake) peak g between t0 and t1, all as positive numbers."""
        i0, i1 = np.searchsorted(self.g_t, t0), np.searchsorted(self.g_t, t1)
        if i1 - i0 < 2:
            return 0.0, 0.0, 0.0, 0.0
        lat, lon = self.g_lat[i0:i1], self.g_lon[i0:i1]
        return (max(0.0, -float(lat.min())), max(0.0, float(lat.max())),
                max(0.0, float(lon.max())), max(0.0, -float(lon.min())))

    def heading_at(self, T, span=0.5):
        # widen the window until the car has moved a few meters (works when slow)
        h = np.array([1.0, 0.0])
        while span <= 20:
            x0, y0, _, _ = self.at(T - span)
            x1, y1, _, _ = self.at(T + span)
            v = np.array([x1 - x0, y1 - y0])
            n = np.linalg.norm(v)
            if n >= 8:
                return v / n
            if n > 0:
                h = v / n
            span *= 2
        return h


# ----------------------------------------------------------------------------
# Formatting / time parsing helpers
# ----------------------------------------------------------------------------

def fmt_lap(t):
    if t is None:
        return "--:--.--"
    m = int(t // 60)
    s = t - 60 * m
    return f"{m}:{s:05.2f}"


def fmt_sector(t):
    if t is None:
        return "--.--"
    return f"{t:.2f}" if t < 60 else fmt_lap(t)


def fmt_clock(t):
    t = max(0, t)
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h}:{m:02d}:{s:04.1f}" if h else f"{m}:{s:04.1f}"


def fmt_hms(t):
    t = max(0, int(round(t)))
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"


def fmt_chapter(t, durations):
    """combined time -> 'N@m:ss' (which file, and how far into it)."""
    if not durations:
        return fmt_clock(t)
    base = 0.0
    for i, d in enumerate(durations):
        if t < base + d or i == len(durations) - 1:
            return f"{i + 1}@{fmt_clock(t - base)}"
        base += d
    return fmt_clock(t)


def fmt_delta(d):
    return f"{'+' if d >= 0 else '-'}{abs(d):.2f}"


def parse_time(s, durations=None):
    """Accepts: 1155 | 19:15 | 1:02:30 | 2@45 | 2@0:45  (chapter@time, chapters numbered from 1
    in the order the files were given).  Returns seconds on the combined timeline."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    base = 0.0
    if "@" in s:
        ch, s = s.split("@", 1)
        ch = int(ch)
        if durations is None or not 1 <= ch <= len(durations):
            raise SystemExit(f"Chapter {ch} in '{ch}@{s}' doesn't exist (you gave {len(durations or [])} file(s)).")
        base = sum(durations[:ch - 1])
    parts = s.split(":")
    if len(parts) > 3 or not all(p.strip() for p in parts):
        raise SystemExit(f"Can't read time '{s}'. Use seconds, mm:ss, h:mm:ss or chapter@time (e.g. 2@0:45).")
    secs = 0.0
    for p in parts:
        secs = secs * 60 + float(p)
    return base + secs


# ----------------------------------------------------------------------------
# Lap detection
# ----------------------------------------------------------------------------

def _line_crossings(t, x, y, a, b, h):
    """Interpolated times at which the path (t, x, y) crosses segment a-b while
    moving in direction h."""
    ab = b - a
    px, py = x[:-1], y[:-1]
    qx, qy = x[1:] - px, y[1:] - py
    fwd = qx * h[0] + qy * h[1] > 0
    den = qx * ab[1] - qy * ab[0]
    ok = fwd & (np.abs(den) > 1e-12)
    den = np.where(ok, den, 1.0)
    apx, apy = a[0] - px, a[1] - py
    u = (apx * ab[1] - apy * ab[0]) / den
    v = (apx * qy - apy * qx) / den
    hit = ok & (u >= 0) & (u < 1) & (v >= 0) & (v <= 1)
    idx = np.nonzero(hit)[0]
    return t[idx] + u[idx] * (t[idx + 1] - t[idx])


def _make_line(center, heading, width):
    c = np.asarray(center, dtype=float)
    h = np.asarray(heading, dtype=float)
    h = h / np.linalg.norm(h)
    perp = np.array([-h[1], h[0]])
    return c - perp * width / 2, c + perp * width / 2, h


class Laps:
    def __init__(self, tel, center_xy, heading, width=30.0, min_lap=20.0, green=None, checkered=None,
                 n_sectors=3):
        self.tel = tel
        self.width = width
        self.sf_a, self.sf_b, self.heading = _make_line(center_xy, heading, width)
        self.green, self.checkered = green, checkered
        self.sf_t = None
        self.drivers = []

        t, x, y = tel.t, tel.x, tel.y
        raw = _line_crossings(t, x, y, self.sf_a, self.sf_b, self.heading)
        self.n_raw = len(raw)
        self.n_pre_green = int(np.sum(raw < green)) if green is not None else 0
        crossings = []
        for ct in raw:
            if green is not None and ct < green:
                continue
            if not crossings or ct - crossings[-1] >= min_lap:
                crossings.append(ct)
                if checkered is not None and ct >= checkered - 15:
                    break
        self.crossings = crossings
        self.times = [crossings[k] - crossings[k - 1] for k in range(1, len(crossings))]
        self.lap_dist = []
        self.profiles = []          # per lap: (distance-into-lap, time-into-lap)
        for k in range(1, len(crossings)):
            c0, c1 = crossings[k - 1], crossings[k]
            idx = np.where((t > c0) & (t < c1))[0]
            d0 = np.interp(c0, t, tel.dist)
            d1 = np.interp(c1, t, tel.dist)
            dd = np.concatenate([[0.0], tel.dist[idx] - d0, [d1 - d0]])
            tt = np.concatenate([[0.0], t[idx] - c0, [c1 - c0]])
            self.profiles.append((np.maximum.accumulate(dd), tt))
            self.lap_dist.append(d1 - d0)
        self._build_sectors(n_sectors)

    # -- construction helpers ---------------------------------------------
    @classmethod
    def from_point(cls, tel, lat, lon, **kw):
        cx, cy = tel.to_xy(lat, lon)
        cx, cy = float(cx), float(cy)
        d = np.hypot(tel.x - cx, tel.y - cy)
        i = int(np.argmin(d))
        if d[i] > 50:
            print(f"WARNING: start/finish point is {d[i]:.0f} m from the nearest GPS point", file=sys.stderr)
        return cls(tel, (cx, cy), tel.heading_at(tel.t[i]), **kw)

    @classmethod
    def from_time(cls, tel, T, **kw):
        x, y, _, _ = tel.at(T)
        L = cls(tel, (x, y), tel.heading_at(T), **kw)
        L.sf_t = T
        return L

    @classmethod
    def auto(cls, tel, **kw):
        # No line given: try the fastest spots (using speed from positions, which
        # ignores GPS speed glitches) and keep the one the car crosses most often.
        dt = np.maximum(np.gradient(tel.t), 1e-3)
        vpos = smooth(np.hypot(np.gradient(tel.x), np.gradient(tel.y)) / dt, 9)
        v = np.minimum(vpos, tel.speed)
        cands = []
        for i in np.argsort(v)[::-1]:
            if all(np.hypot(tel.x[i] - tel.x[j], tel.y[i] - tel.y[j]) > 150 for j in cands):
                cands.append(i)
            if len(cands) >= 12:
                break
        best = None
        for i in cands:
            L = cls(tel, (tel.x[i], tel.y[i]), tel.heading_at(tel.t[i]), **kw)
            if best is None or len(L.crossings) > len(best.crossings):
                best = L
        return best

    # -- sectors -------------------------------------------------------------
    def _build_sectors(self, n):
        self.n_sectors = max(1, n)
        self.sector_times = [[None] * self.n_sectors for _ in self.times]
        self.sector_lines = []
        self.sector_cross = []
        if n < 2 or len(self.times) < 1:
            return
        tel = self.tel
        ref = int(np.argmin(self.times))                     # best lap defines the sector points
        c0 = self.crossings[ref]
        dd, tt = self.profiles[ref]
        for k in range(1, n):
            tk = c0 + float(np.interp(dd[-1] * k / n, dd, tt))
            x, y, _, _ = tel.at(tk)
            a, b, h = _make_line((x, y), tel.heading_at(tk), self.width)
            self.sector_lines.append((a, b, h))
            self.sector_cross.append(_line_crossings(tel.t, tel.x, tel.y, a, b, h))
        for i in range(len(self.times)):
            self.sector_times[i] = self._sector_times_for(i)

    def sector_bounds(self, i, upto=None):
        """Times of the sector boundaries inside lap i (None where not yet crossed)."""
        start = self.crossings[i]
        end = self.crossings[i + 1] if i + 1 < len(self.crossings) else float("inf")
        if upto is not None:
            end = min(end, upto)
        out, prev = [], start
        for cr in self.sector_cross:
            c = cr[(cr > prev + 1.0) & (cr < end)]
            b = float(c[0]) if len(c) else None
            out.append(b)
            if b is not None:
                prev = b
        return out

    def _sector_times_for(self, i):
        b = self.sector_bounds(i)
        pts = [self.crossings[i]] + b + [self.crossings[i + 1]]
        out = []
        for k in range(self.n_sectors):
            p0, p1 = pts[k], pts[k + 1]
            out.append(p1 - p0 if (p0 is not None and p1 is not None and p1 > p0) else None)
        return out

    def best_sectors(self, n_done, driver=None):
        """Best time for each sector over the first n_done laps (optionally one driver's)."""
        out = []
        for k in range(self.n_sectors):
            vals = [self.sector_times[i][k] for i in range(n_done)
                    if self.sector_times[i][k] is not None and (driver is None or self.driver_of_lap(i) == driver)]
            out.append(min(vals) if vals else None)
        return out

    # -- drivers -------------------------------------------------------------
    def set_drivers(self, specs, durations=None):
        """specs: list of 'NAME', 'NAME@lap38', 'NAME@2@12:40', 'NAME@1155' ..."""
        self.drivers = []
        for k, spec in enumerate(specs):
            name, _, when = spec.partition("@")
            name = name.strip()
            if not name:
                raise SystemExit(f"--driver '{spec}': missing a name")
            if not when.strip():
                if k != 0:
                    raise SystemExit(f"--driver '{spec}': only the first driver can omit @when")
                t0 = -1e9
            elif when.strip().lower().startswith("lap"):
                nlap = int(when.strip()[3:].strip())
                if not 1 <= nlap <= len(self.crossings):
                    raise SystemExit(f"--driver '{spec}': lap {nlap} doesn't exist ({len(self.times)} laps found)")
                t0 = self.crossings[nlap - 1]
            else:
                t0 = parse_time(when, durations)
            self.drivers.append((t0, name))
        self.drivers.sort(key=lambda p: p[0])

    def driver_at(self, T):
        if not self.drivers:
            return None
        name = self.drivers[0][1]
        for t0, n in self.drivers:
            if t0 <= T:
                name = n
        return name

    def driver_of_lap(self, i):
        return self.driver_at(self.crossings[i] + 0.01)

    # -- diagnostics -----------------------------------------------------------
    def diagnose(self):
        c = (self.sf_a + self.sf_b) / 2
        d = np.hypot(self.tel.x - c[0], self.tel.y - c[1])
        near = d < 40
        passes = int(np.sum(near[1:] & ~near[:-1]) + (1 if near[0] else 0))
        return float(d.min()), passes

    def explain(self, conv=MS_TO_MPH, unit="mph"):
        lines = [f"No complete laps detected ({self.n_raw} line crossing{'s' if self.n_raw != 1 else ''}"
                 + (f", {self.n_pre_green} of them before --green" if self.n_pre_green else "") + ")."]
        dmin, passes = self.diagnose()
        if self.sf_t is not None:
            _, _, sp, _ = self.tel.at(self.sf_t)
            lines.append(f"  At --sf-time {fmt_clock(self.sf_t)} the car was doing {sp * conv:.0f} {unit}.")
            if sp * conv < 5:
                lines.append("  That's stationary. The timing line has to go where the car drives past at speed:")
                lines.append("  pick a moment when it's crossing the start/finish line on a flying lap.")
        lines.append(f"  The car came within {dmin:.0f} m of the timing line and passed near it {passes} time(s).")
        if dmin > 20:
            lines.append("  The line isn't on the car's path - check --sf-time / --sf-point.")
        elif self.n_pre_green and not self.crossings:
            lines.append("  All crossings were before --green. Is --green too late?")
        elif passes <= 1 and self.sf_t is None:
            lines.append("  The car only visited that spot once, so it isn't a lap.")
        return "\n".join(lines)

    # -- live state ------------------------------------------------------------
    def state(self, T, hold=3.0):
        """Everything the renderer needs at video time T."""
        cr = self.crossings
        n_cross = bisect.bisect_right(cr, T)
        s = {"lap": None, "elapsed": None, "last": None, "best": None, "best_idx": None,
             "delta": None, "completed": n_cross - 1 if n_cross > 0 else 0,
             "hold": None, "new_best": None, "driver": self.driver_at(T), "cur": None,
             "sectors": None, "pred": None, "theo": None, "ghost": None, "d_in_lap": None}
        if n_cross == 0:
            s["lap"] = "PACE" if self.green is not None else "OUT"
            return s
        if self.checkered is not None and n_cross == len(cr) and len(self.times) > 0:
            s["lap"] = "FINISH"
            s["last"] = self.times[-1]
            bi = int(np.argmin(self.times))
            s["best"], s["best_idx"] = self.times[bi], bi
            s["completed"] = len(self.times)
            s["theo"] = self._theo(len(self.times))
            return s
        cur = n_cross - 1                         # index of the lap in progress
        s["lap"] = n_cross
        s["cur"] = cur
        s["elapsed"] = T - cr[cur]
        done = cur
        d_now = float(np.interp(T, self.tel.t, self.tel.dist) - np.interp(cr[cur], self.tel.t, self.tel.dist))
        s["d_in_lap"] = d_now
        if done >= 1:
            s["last"] = self.times[done - 1]
            bi = int(np.argmin(self.times[:done]))
            s["best"], s["best_idx"] = self.times[bi], bi
            if s["elapsed"] < hold:
                li = done - 1
                prev_best = min(self.times[:li]) if li > 0 else None
                s["hold"] = (li, self.times[li], prev_best is None or self.times[li] < prev_best)
            li = done - 1
            if li >= 1 and self.times[li] < min(self.times[:li]):
                s["new_best"] = (li, self.times[li], self.times[li] - min(self.times[:li]), s["elapsed"])
            dd, tt = self.profiles[bi]
            if d_now <= dd[-1]:
                s["delta"] = s["elapsed"] - float(np.interp(d_now, dd, tt))
                s["pred"] = s["best"] + s["delta"]
            if s["elapsed"] < self.times[bi]:
                gx, gy, _, _ = self.tel.at(cr[bi] + s["elapsed"])
                s["ghost"] = (gx, gy)
            s["theo"] = self._theo(done)
        # sectors of the lap in progress
        if self.n_sectors >= 2 and self.sector_lines:
            bounds = self.sector_bounds(cur, upto=T)
            pts = [cr[cur]] + bounds
            best = self.best_sectors(done)
            who = self.driver_at(T)
            personal = self.best_sectors(done, driver=who) if who else best
            secs = []
            for k in range(self.n_sectors):
                p0 = pts[k] if k < len(pts) else None
                p1 = pts[k + 1] if k + 1 < len(pts) else None
                if p0 is None:
                    secs.append(("none", None))
                elif p1 is None:
                    secs.append(("live", T - p0) if all(p is not None for p in pts[:k + 1]) else ("none", None))
                else:
                    v = p1 - p0
                    if best[k] is None or v < best[k]:
                        col = "purple"                     # fastest of the session
                    elif personal[k] is None or v < personal[k]:
                        col = "green"                      # this driver's personal best
                    else:
                        col = "yellow"                     # slower than personal best
                    secs.append((col, v))
            s["sectors"] = secs
        return s

    def _theo(self, n_done):
        if self.n_sectors < 2 or n_done < 1:
            return None
        b = self.best_sectors(n_done)
        return sum(b) if all(v is not None for v in b) else None


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


class Fonts:
    def __init__(self, path=None):
        self.path = None
        for c in ([path] if path else []) + FONT_CANDIDATES:
            try:
                ImageFont.truetype(c, 12)
                self.path = c
                break
            except Exception:
                continue
        if self.path is None:
            print("WARNING: no TrueType font found, using Pillow default (pass --font)", file=sys.stderr)
        self.cache = {}

    def __call__(self, size):
        size = max(6, int(round(size)))
        if size not in self.cache:
            if self.path:
                self.cache[size] = ImageFont.truetype(self.path, size)
            else:
                self.cache[size] = ImageFont.load_default(size=size)
        return self.cache[size]


WHITE = (255, 255, 255, 255)
GREY = (175, 180, 190, 255)
DIM = (255, 255, 255, 55)
PANEL = (10, 12, 16, 165)
PURPLE = (178, 92, 255, 255)
GREEN = (60, 220, 120, 255)
RED = (255, 72, 72, 255)
ACCENT = (255, 196, 0, 255)
GHOST = (255, 255, 255, 150)
YELLOW = (255, 220, 70, 255)
SECTOR_COL = {"white": WHITE, "yellow": YELLOW, "purple": PURPLE, "green": GREEN, "live": GREY,
              "none": (160, 165, 175, 255)}

ALL_ELEMENTS = ["laps", "list", "sectors", "theo", "pred", "clock", "map", "ghost", "elev",
                "speedo", "max", "gmeter", "brake", "trace", "banner", "driver"]


class Renderer:
    SS = 3  # supersampling factor for antialiased shapes

    def __init__(self, tel, laps, W, H, units="mph", speed_max=None, fonts=None,
                 hold=3.0, banner=5.0, show_list=6, hide=(), scale=1.0, g_max=1.5,
                 tz=None, race_start=None, g_peaks="lap", logo=None, sponsors=(), sponsor_slots=0,
                 logo_height=80, team=None, car=None, number=None, brake_g=0.5):
        self.tel, self.laps = tel, laps
        self.brake_g = brake_g
        self.g_peaks = g_peaks
        self.team, self.car, self.number = team, car, number
        self.logo, self.sponsors, self.sponsor_slots = logo, list(sponsors), sponsor_slots
        self.logo_height = logo_height
        self.W, self.H = W, H
        self.u = H / 1080.0 * scale
        self.units = units
        self.conv = MS_TO_MPH if units == "mph" else MS_TO_KMH
        self.dist_conv = 1 / 1609.344 if units == "mph" else 1 / 1000.0
        self.dist_unit = "MI" if units == "mph" else "KM"
        self.hold, self.banner_s, self.show_list = hold, banner, show_list
        self.hide = set(hide)
        self.g_max = g_max
        self.tz = tz
        self.race_start = race_start
        self.f = fonts or Fonts()
        vmax = float(np.max(tel.speed)) * self.conv
        step = 20 if units == "mph" else 40
        self.vmax = speed_max or max(step * 3, math.ceil(vmax * 1.08 / step) * step)
        self._tiles = {}
        self._build_static()

    # -- layout helpers -------------------------------------------------------
    def U(self, v):
        return int(round(v * self.u))

    def on(self, name):
        return name not in self.hide

    def _build_static(self):
        U = self.U
        W, H = self.W, self.H
        self.M = U(30)
        M = self.M
        base = Image.new("RGBA", (W, H), (0, 0, 0, 0))

        # --- left column: lap panel, sectors, lap list
        y = M
        self.panel = None
        if self.on("laps"):
            self.panel = (M, y, M + U(370), y + U(236))
            self._rounded(base, self.panel, U(16), PANEL)
            y = self.panel[3] + U(10)
        self.sector_box = None
        if self.on("sectors") and self.laps and self.laps.n_sectors >= 2:
            self.sector_box = (M, y, M + U(370), y + U(86))
            self._rounded(base, self.sector_box, U(16), PANEL)
            y = self.sector_box[3] + U(10)
        self.list_y = y

        # --- top centre: team logo, then the info strip under it
        top_y = M
        self.logo_box = None
        if self.team or self.car or self.number:
            # text header: team name, with the car (and number badge) underneath
            tf, cf, nf = self.f(U(34)), self.f(U(22)), self.f(U(20))
            num = f"#{self.number}" if self.number else None
            tw = tf.getlength(self.team) if self.team else 0
            cw = cf.getlength(self.car) if self.car else 0
            nw = (nf.getlength(num) + U(18)) if num else 0            # badge width
            line2 = cw + (U(12) if (cw and nw) else 0) + nw
            w = int(max(tw, line2) + U(56))
            two = bool(self.team) and line2 > 0
            h = U(52) + (U(30) if two else 0)
            box = ((W - w) // 2, top_y, (W + w) // 2, top_y + h)
            self._rounded(base, box, U(14), PANEL)
            d = ImageDraw.Draw(base)
            cx = W // 2
            y_team = top_y + U(30) if two else top_y + h // 2
            y_car = top_y + U(62) if two else top_y + h // 2
            if self.team:
                d.text((cx, y_team), self.team, font=tf, fill=WHITE, anchor="mm")
            if line2:
                x = cx - line2 / 2
                if cw:
                    d.text((x, y_car), self.car, font=cf, fill=ACCENT, anchor="lm")
                    x += cw + U(12)
                if num:
                    bh = U(26)
                    nb = (int(x), int(y_car - bh / 2), int(x + nw), int(y_car + bh / 2))
                    self._rounded(base, nb, U(6), ACCENT)
                    d = ImageDraw.Draw(base)
                    d.text(((nb[0] + nb[2]) / 2, y_car), num, font=nf, fill=(14, 16, 20, 255), anchor="mm")
            self.logo_box = box
            top_y += h + U(10)
        if self.logo:
            img = self._load_image(self.logo)
            if img is not None:
                lh = U(self.logo_height)
                max_w = W - 2 * (U(370) + M + U(24))          # between the lap panel and the map
                img = self._fit(img, max_w, lh)
                lx = (W - img.width) // 2
                base.alpha_composite(img, (lx, top_y))
                self.logo_box = (lx, top_y, lx + img.width, top_y + img.height)
                top_y += img.height + U(10)
        self.info_box = None
        if self.on("clock"):
            n_items = (2 if (self.race_start is not None or (self.laps and self.laps.crossings)) else 0) \
                + (1 if self.tel.utc_offset is not None else 0)
            w = U(190) * max(n_items, 1) + U(20)
            self.info_box = ((W - w) // 2, top_y, (W + w) // 2, top_y + U(44))
            self._rounded(base, self.info_box, U(12), PANEL)
            top_y += U(44) + U(10)
        self.banner_y = top_y

        # --- right column: map + elevation
        self.map_box = None
        if self.on("map"):
            ms = U(300)
            self.map_box = (W - M - ms, M, W - M, M + ms)
            self._rounded(base, self.map_box, U(16), PANEL)
            self._build_map(base)
        self.elev_box = None
        if self.on("elev") and self.tel.alt is not None and self.laps and self.laps.times:
            top = (self.map_box[3] + U(10)) if self.map_box else M
            self.elev_box = (W - M - U(300), top, W - M, top + U(84))
            self._rounded(base, self.elev_box, U(12), PANEL)

        # --- bottom right: speedo
        self.dial_c = None
        if self.on("speedo"):
            r = U(150)
            self.dial_r = r
            self.dial_c = (W - M - r, H - M - r)
            self._build_dial(base)
        right_edge = (self.dial_c[0] - self.dial_r - U(24)) if self.dial_c else (W - M)

        # --- G-meter + brake light, left of the speedo
        self.g_c = None
        if self.on("gmeter"):
            rg = U(105)
            self.g_r = rg
            self.g_c = (right_edge - rg, H - M - rg)
            self._build_gmeter(base)
            right_edge = self.g_c[0] - rg - U(24)
        self.brake_box = None
        if self.on("brake"):
            if self.g_c:
                cx = self.g_c[0]
                top = self.g_c[1] - self.g_r - U(48)
            else:
                cx = right_edge - U(70)
                top = H - M - U(36)
            self.brake_box = (cx - U(62), top, cx + U(62), top + U(36))

        # --- bottom left: lap history (anchored to the bottom edge)
        self.list_w = U(370)
        list_right = M
        if self.on("list") and self.show_list and self.laps and self.laps.times:
            list_right = M + self.list_w + U(16)

        # --- bottom row, right to left: speedo, G-meter, then the speed trace next to it
        self.trace_box = None
        if self.on("trace") and self.laps and self.laps.times:
            w = min(U(480), right_edge - list_right - U(10))
            if w > U(200):
                self.trace_box = (right_edge - w, H - M - U(140), right_edge, H - M)
                self._rounded(base, self.trace_box, U(12), PANEL)
                d = ImageDraw.Draw(base)
                d.text((self.trace_box[0] + U(12), self.trace_box[1] + U(8)), "SPEED  vs best lap",
                       font=self.f(U(15)), fill=GREY)
                right_edge = self.trace_box[0] - U(16)

        # --- bottom centre: sponsor logos between the lap history and the speed trace
        self.sponsor_boxes = []
        n_slots = max(self.sponsor_slots, len(self.sponsors))
        if n_slots:
            left = list_right
            right = right_edge
            gap = U(14)
            sh = U(110)
            avail = right - left - gap * (n_slots - 1)
            sw = avail // n_slots
            if sw > U(80):
                y1 = H - M
                y0 = y1 - sh
                for k in range(n_slots):
                    box = (left + k * (sw + gap), y0, left + k * (sw + gap) + sw, y1)
                    self.sponsor_boxes.append(box)
                    self._rounded(base, box, U(12), PANEL)
                    img = self._load_image(self.sponsors[k]) if k < len(self.sponsors) else None
                    if img is not None:
                        img = self._fit(img, sw - U(24), sh - U(20))
                        base.alpha_composite(img, (box[0] + (sw - img.width) // 2, y0 + (sh - img.height) // 2))
                    else:
                        d = ImageDraw.Draw(base)
                        inset = U(10)
                        self._dashed_rect(d, (box[0] + inset, y0 + inset, box[2] - inset, y1 - inset), U(8), U(6))
                        d.text(((box[0] + box[2]) // 2, (y0 + y1) // 2), "SPONSOR", font=self.f(U(22)),
                               fill=(255, 255, 255, 90), anchor="mm")
            else:
                print("WARNING: no room for sponsor logos at this layout; try --scale 0.9 or hide something",
                      file=sys.stderr)

        # sprites
        self.dot = self._dot_sprite(U(10), ACCENT, (0, 0, 0, 255), U(3))
        self.ghost_dot = self._dot_sprite(U(8), GHOST, (0, 0, 0, 120), U(2))
        self.g_dot = self._dot_sprite(U(9), ACCENT, (0, 0, 0, 255), U(2))
        self.g_trail = self._dot_sprite(U(5), (255, 196, 0, 110), None, 0)
        self.static = base

    def _load_image(self, path):
        try:
            return Image.open(path).convert("RGBA")
        except Exception as e:
            print(f"WARNING: couldn't load image {path}: {e}", file=sys.stderr)
            return None

    @staticmethod
    def _fit(img, max_w, max_h):
        sc = min(max_w / img.width, max_h / img.height)
        if sc < 1 or sc > 1.001:
            img = img.resize((max(1, int(img.width * sc)), max(1, int(img.height * sc))), Image.LANCZOS)
        return img

    @staticmethod
    def _dashed_rect(d, box, dash, gap):
        x0, y0, x1, y1 = box
        col = (255, 255, 255, 90)
        for (ax, ay, bx, by) in ((x0, y0, x1, y0), (x0, y1, x1, y1), (x0, y0, x0, y1), (x1, y0, x1, y1)):
            length = max(bx - ax, by - ay)
            pos = 0
            while pos < length:
                e = min(pos + dash, length)
                if bx > ax:
                    d.line([(ax + pos, ay), (ax + e, ay)], fill=col, width=2)
                else:
                    d.line([(ax, ay + pos), (ax, ay + e)], fill=col, width=2)
                pos += dash + gap

    def _dot_sprite(self, r, fill, outline, width):
        S = self.SS
        spr = Image.new("RGBA", ((2 * r + 4) * S, (2 * r + 4) * S), (0, 0, 0, 0))
        ImageDraw.Draw(spr).ellipse((2 * S, 2 * S, (2 * r + 2) * S, (2 * r + 2) * S),
                                    fill=fill, outline=outline, width=width * S)
        return spr.resize((2 * r + 4, 2 * r + 4), Image.LANCZOS)

    def _rounded(self, img, box, rad, fill):
        S = self.SS
        x0, y0, x1, y1 = box
        tile = Image.new("RGBA", ((x1 - x0) * S, (y1 - y0) * S), (0, 0, 0, 0))
        ImageDraw.Draw(tile).rounded_rectangle((0, 0, tile.width - 1, tile.height - 1), rad * S, fill=fill)
        img.alpha_composite(tile.resize((x1 - x0, y1 - y0), Image.LANCZOS), (x0, y0))

    def _panel_tile(self, box, rad=None, fill=PANEL):
        key = ("panel", box, fill)
        if key not in self._tiles:
            t = Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), (0, 0, 0, 0))
            self._rounded(t, (0, 0, box[2] - box[0], box[3] - box[1]), rad or self.U(16), fill)
            self._tiles[key] = t
        return self._tiles[key]

    # -- static pieces ---------------------------------------------------------
    def _build_map(self, base):
        S = self.SS
        tel, laps = self.tel, self.laps
        if laps and len(laps.crossings) >= 2:
            m = (tel.t >= laps.crossings[0]) & (tel.t <= laps.crossings[-1])
        else:
            m = np.ones_like(tel.t, dtype=bool)
        xs, ys = tel.x[m], tel.y[m]
        x0, y0, x1, y1 = self.map_box
        pad = self.U(28)
        bw, bh = (x1 - x0) - 2 * pad, (y1 - y0) - 2 * pad
        minx, maxx, miny, maxy = xs.min(), xs.max(), ys.min(), ys.max()
        sc = min(bw / max(maxx - minx, 1), bh / max(maxy - miny, 1))
        ox = x0 + pad + (bw - (maxx - minx) * sc) / 2
        oy = y0 + pad + (bh - (maxy - miny) * sc) / 2
        self.map_tf = lambda X, Y: (ox + (X - minx) * sc, oy + (maxy - Y) * sc)

        w, h = x1 - x0, y1 - y0
        tile = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
        d = ImageDraw.Draw(tile)
        step = max(1, len(xs) // 4000)
        pts = [self.map_tf(X, Y) for X, Y in zip(xs[::step], ys[::step])]
        pts = [((px - x0) * S, (py - y0) * S) for px, py in pts]
        d.line(pts, fill=(255, 255, 255, 70), width=self.U(11) * S, joint="curve")
        d.line(pts, fill=(255, 255, 255, 215), width=self.U(4) * S, joint="curve")
        if laps:
            def mark(a, b, col, ln):
                pa, pb = self.map_tf(*a), self.map_tf(*b)
                mid = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
                v = np.array([pb[0] - pa[0], pb[1] - pa[1]])
                v = v / (np.linalg.norm(v) or 1) * self.U(ln)
                d.line([((mid[0] - v[0] - x0) * S, (mid[1] - v[1] - y0) * S),
                        ((mid[0] + v[0] - x0) * S, (mid[1] + v[1] - y0) * S)], fill=col, width=self.U(4) * S)
            mark(laps.sf_a, laps.sf_b, RED, 12)
            if self.on("sectors"):
                for a, b, _ in laps.sector_lines:
                    mark(a, b, (255, 255, 255, 160), 8)
        base.alpha_composite(tile.resize((w, h), Image.LANCZOS), (x0, y0))

    def _angle(self, v):
        f = min(max(v / self.vmax, 0.0), 1.0)
        return 135 + 270 * f  # PIL degrees, clockwise from 3 o'clock

    def _build_dial(self, base):
        S = self.SS
        r = self.dial_r
        cx, cy = self.dial_c
        size = 2 * r
        tile = Image.new("RGBA", (size * S, size * S), (0, 0, 0, 0))
        d = ImageDraw.Draw(tile)
        C = r * S
        d.ellipse((0, 0, size * S - 1, size * S - 1), fill=PANEL)
        ra = int(r * 0.86)
        aw = max(4, int(r * 0.085))
        d.arc((C - ra * S, C - ra * S, C + ra * S, C + ra * S), 135, 405, fill=DIM, width=aw * S)
        major = 20 if self.units == "mph" else 40
        if self.vmax <= 80:
            major = 10
        minor = major / 2
        v = 0.0
        while v <= self.vmax + 1e-6:
            a = math.radians(self._angle(v))
            is_major = abs(v / major - round(v / major)) < 1e-6
            r_out = ra - aw / 2 - r * 0.03
            r_in = r_out - (r * 0.085 if is_major else r * 0.045)
            d.line([(C + r_out * S * math.cos(a), C + r_out * S * math.sin(a)),
                    (C + r_in * S * math.cos(a), C + r_in * S * math.sin(a))],
                   fill=WHITE if is_major else GREY, width=max(1, int((r * 0.02 if is_major else r * 0.01) * S)))
            if is_major:
                rl = r_in - r * 0.11
                d.text((C + rl * S * math.cos(a), C + rl * S * math.sin(a)), f"{int(v)}",
                       font=self.f(r * 0.105 * S), fill=GREY, anchor="mm")
            v += minor
        d.text((C, C + r * 0.74 * S), "MPH" if self.units == "mph" else "KM/H",
               font=self.f(r * 0.12 * S), fill=GREY, anchor="mm")
        base.alpha_composite(tile.resize((size, size), Image.LANCZOS), (cx - r, cy - r))
        self.dial_ra, self.dial_aw = ra, aw

    def _build_gmeter(self, base):
        S = self.SS
        rg = self.g_r
        cx, cy = self.g_c
        size = 2 * rg
        tile = Image.new("RGBA", (size * S, size * S), (0, 0, 0, 0))
        d = ImageDraw.Draw(tile)
        C = rg * S
        d.ellipse((0, 0, size * S - 1, size * S - 1), fill=PANEL)
        self.g_scale = rg * 0.86 / self.g_max          # px per g
        ring = 0.5
        while ring < self.g_max + 1e-6:
            rr = ring * self.g_scale * S
            d.ellipse((C - rr, C - rr, C + rr, C + rr), outline=(255, 255, 255, 70 if ring % 1 else 130),
                      width=max(1, int(self.U(1.5) * S)))
            if ring % 1 == 0:
                d.text((C + rr * 0.707 + self.U(4) * S, C - rr * 0.707 - self.U(4) * S), f"{int(ring)}g",
                       font=self.f(self.U(13) * S), fill=GREY, anchor="lb")
            ring += 0.5
        lim = rg * 0.9 * S
        d.line([(C - lim, C), (C + lim, C)], fill=(255, 255, 255, 60), width=max(1, int(self.U(1) * S)))
        d.line([(C, C - lim), (C, C + lim)], fill=(255, 255, 255, 60), width=max(1, int(self.U(1) * S)))
        d.text((C, C - rg * 0.93 * S), "ACCEL", font=self.f(self.U(11) * S), fill=GREY, anchor="mt")
        d.text((C, C + rg * 0.93 * S), "BRAKE", font=self.f(self.U(11) * S), fill=GREY, anchor="mb")
        base.alpha_composite(tile.resize((size, size), Image.LANCZOS), (cx - rg, cy - rg))

    # -- per-frame ---------------------------------------------------------------
    def frame(self, T):
        img = self.static.copy()
        d = ImageDraw.Draw(img)
        x, y, sp, dist = self.tel.at(T)
        in_range = self.tel.t[0] - 1 <= T <= self.tel.t[-1] + 1
        st = self.laps.state(T, self.hold) if self.laps else {"lap": None}

        if self.dial_c:
            self._draw_speedo(img, d, sp * self.conv if in_range else 0.0,
                              float(np.interp(T, self.tel.t, self.tel.speed_max_sofar)) * self.conv)
        if self.g_c or self.brake_box:
            self._draw_gmeter(img, d, T, in_range, st)
        if self.map_box and in_range:
            if self.on("ghost") and st.get("ghost"):
                gx, gy = self.map_tf(*st["ghost"])
                r = self.ghost_dot.width // 2
                img.alpha_composite(self.ghost_dot, (int(round(gx)) - r, int(round(gy)) - r))
            px, py = self.map_tf(x, y)
            r = self.dot.width // 2
            img.alpha_composite(self.dot, (int(round(px)) - r, int(round(py)) - r))
        if self.elev_box:
            self._draw_elev(img, d, st, dist)
        if self.trace_box:
            self._draw_trace(img, d, st, T)
        if self.info_box:
            self._draw_info(img, d, T, dist)
        if self.panel:
            self._draw_laps(img, d, T, st)
        if self.sector_box:
            self._draw_sectors(img, d, st)
        if self.on("list"):
            self._draw_list(img, d, st)
        if self.on("banner"):
            nb = st.get("new_best")
            if nb and nb[3] < self.banner_s:
                li, lt, gain, age = nb
                alpha = 1.0 if age < self.banner_s - 0.6 else max(0.0, (self.banner_s - age) / 0.6)
                who = self.laps.driver_of_lap(li) if self.on("driver") else None
                sub = (f"{who.upper()}  ·  " if who else "") + f"LAP {li + 1}  ·  {fmt_delta(gain)}"
                self._banner(img, f"NEW BEST LAP  {fmt_lap(lt)}", sub, alpha)
        return img

    def _draw_speedo(self, img, d, v, vmax_sofar):
        S, U = 2, self.U
        r, ra, aw = self.dial_r, self.dial_ra, self.dial_aw
        cx, cy = self.dial_c
        size = 2 * r
        C = r * S
        tile = Image.new("RGBA", (size * S, size * S), (0, 0, 0, 0))
        td = ImageDraw.Draw(tile)
        ang = self._angle(v)
        if ang > 135.3:
            td.arc((C - ra * S, C - ra * S, C + ra * S, C + ra * S), 135, ang, fill=ACCENT, width=aw * S)
        if self.on("max") and vmax_sofar > 1:
            am = math.radians(self._angle(vmax_sofar))
            r1, r2 = ra + aw / 2 + r * 0.01, ra + aw / 2 + r * 0.08
            td.line([(C + r1 * S * math.cos(am), C + r1 * S * math.sin(am)),
                     (C + r2 * S * math.cos(am), C + r2 * S * math.sin(am))], fill=RED, width=int(r * 0.03 * S))
        a = math.radians(ang)
        rn = ra - aw / 2 - r * 0.02
        td.line([(C - r * 0.12 * S * math.cos(a), C - r * 0.12 * S * math.sin(a)),
                 (C + rn * S * math.cos(a), C + rn * S * math.sin(a))], fill=ACCENT, width=int(r * 0.032 * S))
        hub = int(r * 0.07 * S)
        td.ellipse((C - hub, C - hub, C + hub, C + hub), fill=(30, 32, 38, 255), outline=ACCENT, width=int(r * 0.016 * S))
        img.alpha_composite(tile.reduce(S), (cx - r, cy - r))
        d.text((cx, cy + r * 0.45), f"{int(round(v))}", font=self.f(r * 0.37), fill=WHITE, anchor="mm")
        if self.on("max") and vmax_sofar > 1:
            d.text((cx, cy - r * 0.42), f"MAX {int(round(vmax_sofar))}", font=self.f(r * 0.11), fill=GREY, anchor="mm")

    def _draw_gmeter(self, img, d, T, in_range, st=None):
        U = self.U
        g_lon, g_lat = self.tel.g_at(T) if in_range else (0.0, 0.0)
        if self.g_c:
            cx, cy = self.g_c
            # peak markers (reset each lap, or whole session with --g-peaks session)
            peaks = None
            if self.g_peaks != "off" and in_range:
                if self.g_peaks == "session":
                    t0 = self.race_start if self.race_start is not None else self.tel.t[0]
                elif st and st.get("cur") is not None:
                    t0 = self.laps.crossings[st["cur"]]
                else:
                    t0 = self.tel.t[0]
                peaks = self.tel.g_peaks(t0, T)
                left, right, acc, brk = peaks
                lw = max(1, U(2))
                ln = U(9)
                for val, dx, dy in ((left, -1, 0), (right, 1, 0), (acc, 0, -1), (brk, 0, 1)):
                    if val < 0.15:
                        continue
                    v = min(val, self.g_max) * self.g_scale
                    px, py = cx + dx * v, cy + dy * v
                    if dx:
                        d.line([(px, py - ln), (px, py + ln)], fill=RED, width=lw)
                    else:
                        d.line([(px - ln, py), (px + ln, py)], fill=RED, width=lw)
            # trail of the last half second
            r = self.g_trail.width // 2
            for k in range(8, 0, -1):
                tl, tt = self.tel.g_at(T - k * 0.06) if in_range else (0.0, 0.0)
                px = cx + max(-self.g_max, min(self.g_max, tt)) * self.g_scale
                py = cy - max(-self.g_max, min(self.g_max, tl)) * self.g_scale
                img.alpha_composite(self.g_trail, (int(round(px)) - r, int(round(py)) - r))
            px = cx + max(-self.g_max, min(self.g_max, g_lat)) * self.g_scale
            py = cy - max(-self.g_max, min(self.g_max, g_lon)) * self.g_scale
            r = self.g_dot.width // 2
            img.alpha_composite(self.g_dot, (int(round(px)) - r, int(round(py)) - r))
            mag = math.hypot(g_lon, g_lat)
            d.text((cx, cy + self.g_r * 0.62), f"{mag:.2f} g", font=self.f(U(20)), fill=WHITE, anchor="mm")
            if peaks:
                left, right, acc, brk = peaks
                lv, rv = f"{max(left, right):.2f}", f"{brk:.2f}"
                ll, rl = "MAX LAT", "MAX BRK"
                col = (255, 120, 120, 255)
            else:
                lv, rv = f"{abs(g_lat):.2f}", f"{abs(g_lon):.2f}"
                ll, rl = "LAT", "LON"
                col = GREY
            d.text((cx - self.g_r * 0.62, cy + self.g_r * 0.36), lv, font=self.f(U(14)), fill=col, anchor="mm")
            d.text((cx + self.g_r * 0.62, cy + self.g_r * 0.36), rv, font=self.f(U(14)), fill=col, anchor="mm")
            d.text((cx - self.g_r * 0.62, cy + self.g_r * 0.36 + U(15)), ll, font=self.f(U(9)), fill=GREY, anchor="mm")
            d.text((cx + self.g_r * 0.62, cy + self.g_r * 0.36 + U(15)), rl, font=self.f(U(9)), fill=GREY, anchor="mm")
        if self.brake_box:
            braking = g_lon < -self.brake_g
            box = self.brake_box
            tile = self._panel_tile(box, U(8), (200, 30, 30, 235) if braking else PANEL)
            img.alpha_composite(tile, box[:2])
            d.text(((box[0] + box[2]) // 2, (box[1] + box[3]) // 2), "BRAKE", font=self.f(U(20)),
                   fill=WHITE if braking else (120, 125, 135, 255), anchor="mm")

    def _draw_info(self, img, d, T, dist):
        U = self.U
        x0, y0, x1, y1 = self.info_box
        cy = (y0 + y1) // 2
        parts = []
        rs = self.race_start
        if rs is None and self.laps and self.laps.crossings:
            rs = self.laps.crossings[0]
        if rs is not None:
            if T >= rs:
                d_race = (dist - self.tel.at(rs)[3]) * self.dist_conv
                parts.append(("RACE", fmt_hms(T - rs)))
                parts.append((self.dist_unit, f"{d_race:.1f}"))
            else:
                parts.append(("RACE", "PRE-RACE"))
        if self.tel.utc_offset is not None:
            try:
                ts = self.tel.utc_offset + T
                dt = _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc)
                dt = dt.astimezone(self.tz) if self.tz else dt.astimezone()
                parts.append(("TIME", dt.strftime("%I:%M:%S %p").lstrip("0")))
            except (OverflowError, OSError, ValueError):
                pass
        if not parts:
            return
        # lay out evenly
        seg = (x1 - x0) / len(parts)
        for i, (lab, val) in enumerate(parts):
            cx = x0 + seg * (i + 0.5)
            lw = self.f(U(15)).getlength(lab)
            vw = self.f(U(22)).getlength(val)
            tot = lw + U(8) + vw
            d.text((cx - tot / 2, cy), lab, font=self.f(U(15)), fill=GREY, anchor="lm")
            d.text((cx - tot / 2 + lw + U(8), cy), val, font=self.f(U(22)), fill=WHITE, anchor="lm")

    def _draw_laps(self, img, d, T, st):
        U = self.U
        laps = self.laps
        x0, y0, x1, y1 = self.panel
        L, R = x0 + U(22), x1 - U(22)
        hdr_f, big_f = self.f(U(24)), self.f(U(68))
        hy = y0 + U(18)

        # header row
        right_txt = None
        if st.get("hold"):
            li, lt, is_best = st["hold"]
            lab = f"LAP {li + 1}"
            d.text((L, hy), lab, font=hdr_f, fill=WHITE)
            prev = min(laps.times[:li]) if li > 0 else None
            if prev is not None:
                g = lt - prev
                right_txt = (fmt_delta(g), PURPLE if g < 0 else RED)
        elif st["lap"] == "OUT":
            lab = "OUT LAP"
            d.text((L, hy), lab, font=hdr_f, fill=WHITE)
        elif st["lap"] == "PACE":
            lab = "PACE LAP"
            d.text((L, hy), lab, font=hdr_f, fill=ACCENT)
        elif st["lap"] == "FINISH":
            lab = "CHECKERED"
            d.text((L, hy), lab, font=hdr_f, fill=ACCENT)
            right_txt = (f"{st['completed']} LAPS", GREY)
        elif st["lap"]:
            lab = f"LAP {st['lap']}"
            d.text((L, hy), lab, font=hdr_f, fill=WHITE)
        else:
            lab = "NO LAPS"
            d.text((L, hy), lab, font=hdr_f, fill=GREY)
        if right_txt is None and st.get("delta") is not None:
            dv = st["delta"]
            right_txt = (fmt_delta(dv), GREEN if dv < 0 else RED)
        if right_txt:
            d.text((R, hy - U(1)), right_txt[0], font=self.f(U(26)), fill=right_txt[1], anchor="ra")
        drv = st.get("driver") if self.on("driver") else None
        if drv and st["lap"] != "FINISH":
            xx = L + hdr_f.getlength(lab) + U(14)
            avail = (R - (self.f(U(26)).getlength(right_txt[0]) + U(14) if right_txt else 0)) - xx
            fs = U(20)
            while fs > U(12) and self.f(fs).getlength(drv.upper()) > avail:
                fs -= 1
            if avail > U(30):
                d.text((xx, hy + U(24) - fs - U(1)), drv.upper(), font=self.f(fs), fill=ACCENT)

        # big timer
        by = y0 + U(50)
        if st.get("hold"):
            li, lt, is_best = st["hold"]
            d.text((L, by), fmt_lap(lt), font=big_f, fill=PURPLE if is_best else WHITE)
        elif st["lap"] == "FINISH":
            d.text((L, by), fmt_lap(st["best"]), font=big_f, fill=PURPLE)
        else:
            d.text((L, by), fmt_lap(st.get("elapsed")), font=big_f,
                   fill=WHITE if st.get("elapsed") is not None else GREY)

        # last / best
        ry = y0 + U(172)
        d.text((L, ry), "LAST", font=self.f(U(18)), fill=GREY, anchor="ls")
        d.text((R, ry), fmt_lap(st.get("last")), font=self.f(U(26)), fill=WHITE, anchor="rs")
        ry += U(44)
        d.text((L, ry), "BEST", font=self.f(U(18)), fill=GREY, anchor="ls")
        if self.on("driver") and st.get("best_idx") is not None and laps.driver_of_lap(st["best_idx"]):
            d.text((L + self.f(U(18)).getlength("BEST") + U(10), ry), laps.driver_of_lap(st["best_idx"]).upper(),
                   font=self.f(U(15)), fill=PURPLE, anchor="ls")
        d.text((R, ry), fmt_lap(st.get("best")), font=self.f(U(26)),
               fill=PURPLE if st.get("best") is not None else WHITE, anchor="rs")

    def _draw_sectors(self, img, d, st):
        U = self.U
        x0, y0, x1, y1 = self.sector_box
        n = self.laps.n_sectors
        secs = st.get("sectors") or [("none", None)] * n
        L, R = x0 + U(16), x1 - U(16)
        seg = (R - L) / n
        ty = y0 + U(9)
        for k, (col, v) in enumerate(secs):
            cx = L + seg * (k + 0.5)
            d.text((cx, ty), f"S{k + 1}", font=self.f(U(12)), fill=GREY, anchor="mt")
            d.text((cx, ty + U(15)), fmt_sector(v) if v is not None else "--.--",
                   font=self.f(U(24)), fill=SECTOR_COL[col], anchor="mt")
        # theoretical best / predicted
        ly = y1 - U(11)
        items = []
        if self.on("theo") and st.get("theo") is not None:
            items.append(("THEO", fmt_lap(st["theo"]), PURPLE))
        if self.on("pred") and st.get("pred") is not None and not st.get("hold"):
            items.append(("PRED", fmt_lap(st["pred"]), WHITE))
        if items:
            seg = (R - L) / len(items)
            for i, (lab, val, col) in enumerate(items):
                cx = L + seg * (i + 0.5)
                lw = self.f(U(12)).getlength(lab)
                vw = self.f(U(17)).getlength(val)
                tot = lw + U(6) + vw
                d.text((cx - tot / 2, ly), lab, font=self.f(U(12)), fill=GREY, anchor="ls")
                d.text((cx - tot / 2 + lw + U(6), ly), val, font=self.f(U(17)), fill=col, anchor="ls")

    def _draw_list(self, img, d, st):
        U = self.U
        laps = self.laps
        n = st.get("completed", 0) if laps else 0
        if not (n and self.show_list):
            return
        x0 = self.M
        x1 = x0 + U(370)
        L, R = x0 + U(22), x1 - U(22)
        first = max(0, n - self.show_list)
        rows = list(range(first, n))
        rh = U(31)
        box_h = U(14) + rh * len(rows)
        by0 = self.H - self.M - box_h
        box = (x0, by0, x1, by0 + box_h)
        img.alpha_composite(self._panel_tile(box), box[:2])
        bi = st.get("best_idx")
        for j, li in enumerate(rows):
            yy = by0 + U(9) + rh * j + rh // 2
            col = PURPLE if li == bi else WHITE
            d.text((L, yy), f"L{li + 1}", font=self.f(U(20)), fill=GREY if li != bi else PURPLE, anchor="lm")
            who = laps.driver_of_lap(li) if self.on("driver") else None
            if who:
                d.text((L + U(66), yy), who.upper()[:12], font=self.f(U(16)),
                       fill=GREY if li != bi else PURPLE, anchor="lm")
            d.text((R, yy), fmt_lap(laps.times[li]), font=self.f(U(22)), fill=col, anchor="rm")

    def _elev_tile(self, bi):
        key = ("elev", bi)
        if key in self._tiles:
            return self._tiles[key]
        S, U = 2, self.U
        x0, y0, x1, y1 = self.elev_box
        w, h = x1 - x0, y1 - y0
        tel, laps = self.tel, self.laps
        c0, c1 = laps.crossings[bi], laps.crossings[bi + 1]
        m = (tel.t >= c0) & (tel.t <= c1)
        dd = tel.dist[m] - tel.dist[m][0]
        aa = tel.alt[m]
        allalt = tel.alt[(tel.t >= laps.crossings[0]) & (tel.t <= laps.crossings[-1])]
        lo, hi = float(np.percentile(allalt, 1)), float(np.percentile(allalt, 99))
        if hi - lo < 5:
            lo, hi = (lo + hi) / 2 - 2.5, (lo + hi) / 2 + 2.5
        padx, padt, padb = U(12), U(22), U(10)
        gw, gh = w - 2 * padx, h - padt - padb
        L = max(dd[-1], 1.0)
        tile = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
        d = ImageDraw.Draw(tile)
        pts = [((padx + dv / L * gw) * S, (padt + (1 - (av - lo) / (hi - lo)) * gh) * S) for dv, av in zip(dd, aa)]
        poly = [((padx) * S, (padt + gh) * S)] + pts + [((padx + gw) * S, (padt + gh) * S)]
        d.polygon(poly, fill=(255, 255, 255, 45))
        d.line(pts, fill=(255, 255, 255, 200), width=U(2) * S, joint="curve")
        tile = tile.resize((w, h), Image.LANCZOS)
        td = ImageDraw.Draw(tile)
        conv, unit = (3.28084, "ft") if self.units == "mph" else (1.0, "m")
        td.text((padx, U(6)), "ELEVATION", font=self.f(U(13)), fill=GREY)
        td.text((w - padx, U(6)), f"{(hi - lo) * conv:.0f} {unit} range", font=self.f(U(13)), fill=GREY, anchor="ra")
        self._tiles[key] = (tile, (padx, padt, gw, gh, L, lo, hi))
        return self._tiles[key]

    def _draw_elev(self, img, d, st, dist):
        bi = st.get("best_idx")
        if bi is None:
            bi = int(np.argmin(self.laps.times))
        tile, (padx, padt, gw, gh, L, lo, hi) = self._elev_tile(bi)
        x0, y0, x1, y1 = self.elev_box
        img.alpha_composite(tile, (x0, y0))
        d_in = st.get("d_in_lap")
        if d_in is not None and 0 <= d_in <= L * 1.02:
            px = x0 + padx + min(d_in, L) / L * gw
            T = None
            alt = float(np.interp(dist, self.tel.dist, self.tel.alt))
            py = y0 + padt + (1 - (alt - lo) / (hi - lo)) * gh
            py = min(max(py, y0 + padt), y0 + padt + gh)
            r = self.ghost_dot.width // 2
            img.alpha_composite(self.dot, (int(round(px)) - self.dot.width // 2, int(round(py)) - self.dot.width // 2))

    def _trace_ref(self, bi):
        key = ("trace", bi)
        if key in self._tiles:
            return self._tiles[key]
        S, U = 2, self.U
        x0, y0, x1, y1 = self.trace_box
        w, h = x1 - x0, y1 - y0
        padx, padt, padb = U(12), U(28), U(10)
        gw, gh = w - 2 * padx, h - padt - padb
        tel, laps = self.tel, self.laps
        c0, c1 = laps.crossings[bi], laps.crossings[bi + 1]
        m = (tel.t >= c0) & (tel.t <= c1)
        dd = tel.dist[m] - tel.dist[m][0]
        vv = tel.speed[m] * self.conv
        L = max(dd[-1], 1.0)
        tile = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
        d = ImageDraw.Draw(tile)
        pts = [((padx + dv / L * gw) * S, (padt + (1 - min(sv / self.vmax, 1)) * gh) * S) for dv, sv in zip(dd, vv)]
        d.line(pts, fill=(255, 255, 255, 170), width=U(2) * S, joint="curve")
        tile = tile.resize((w, h), Image.LANCZOS)
        self._tiles[key] = (tile, (padx, padt, gw, gh, L))
        return self._tiles[key]

    def _draw_trace(self, img, d, st, T):
        S, U = 2, self.U
        bi = st.get("best_idx")
        cur = st.get("cur")
        if bi is None or cur is None:
            return
        tile, (padx, padt, gw, gh, L) = self._trace_ref(bi)
        x0, y0, x1, y1 = self.trace_box
        img.alpha_composite(tile, (x0, y0))
        tel = self.tel
        c0 = self.laps.crossings[cur]
        m = (tel.t >= c0) & (tel.t <= T)
        if m.sum() < 2:
            return
        dd = tel.dist[m] - float(np.interp(c0, tel.t, tel.dist))
        vv = tel.speed[m] * self.conv
        keep = dd <= L * 1.02
        dd, vv = dd[keep], vv[keep]
        if len(dd) < 2:
            return
        step = max(1, len(dd) // 250)
        if step > 1:
            dd, vv = np.append(dd[::step], dd[-1]), np.append(vv[::step], vv[-1])
        w, h = x1 - x0, y1 - y0
        live = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
        ld = ImageDraw.Draw(live)
        pts = [((padx + min(dv, L) / L * gw) * S, (padt + (1 - min(sv / self.vmax, 1)) * gh) * S)
               for dv, sv in zip(dd, vv)]
        ld.line(pts, fill=ACCENT, width=U(3) * S, joint="curve")
        img.alpha_composite(live.reduce(S), (x0, y0))
        px, py = x0 + pts[-1][0] / S, y0 + pts[-1][1] / S
        img.alpha_composite(self.ghost_dot, (int(round(px)) - self.ghost_dot.width // 2,
                                             int(round(py)) - self.ghost_dot.width // 2))

    def _banner(self, img, text, sub, alpha):
        U = self.U
        w, h = U(600), U(104)
        x0 = (self.W - w) // 2
        y0 = self.banner_y if self.logo_box else self.M
        key = ("banner", text, sub)
        if key not in self._tiles:
            t = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            self._rounded(t, (0, 0, w, h), U(16), (120, 40, 210, 235))
            td = ImageDraw.Draw(t)
            td.text((w // 2, U(38)), text, font=self.f(U(36)), fill=WHITE, anchor="mm")
            td.text((w // 2, U(78)), sub, font=self.f(U(22)), fill=(235, 220, 255, 255), anchor="mm")
            self._tiles[key] = t
        tile = self._tiles[key]
        if alpha < 1.0:
            tile = tile.copy()
            tile.putalpha(tile.getchannel("A").point(lambda a: int(a * alpha)))
        img.alpha_composite(tile, (x0, y0))


# ----------------------------------------------------------------------------
# Video helpers
# ----------------------------------------------------------------------------

def need(tool):
    if shutil.which(tool) is None:
        raise SystemExit(f"'{tool}' not found. Install FFmpeg and make sure it's on your PATH.")


def probe(path):
    need("ffprobe")
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate", "-of", "csv=p=0:s=,", path],
        capture_output=True, text=True, check=True).stdout.strip().splitlines()[0].split(",")
    w, h = int(out[0]), int(out[1])
    num, den = out[2].split("/")
    return w, h, float(num) / float(den)


def probe_color(path):
    """Colour tags of the source video (range, matrix, primaries, transfer), with
    HD defaults for anything the file doesn't say. Used to tag the overlay
    stream identically so ffmpeg never converts the picture between spaces."""
    tags = {"range": "tv", "space": "bt709", "primaries": "bt709", "trc": "bt709"}
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=color_range,color_space,color_primaries,color_transfer", "-of", "default=nw=1", path],
            capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            k, _, v = line.partition("=")
            v = v.strip()
            if not v or v == "unknown":
                continue
            if k == "color_range":
                tags["range"] = v
            elif k == "color_space":
                tags["space"] = v
            elif k == "color_primaries":
                tags["primaries"] = v
            elif k == "color_transfer":
                tags["trc"] = v
    except (subprocess.CalledProcessError, OSError, IndexError):
        pass
    return tags


def color_args(tags):
    return ["-color_range", tags["range"], "-colorspace", tags["space"],
            "-color_primaries", tags["primaries"], "-color_trc", tags["trc"]]


def lrv_for(mp4):
    d, b = os.path.split(mp4)
    stem = os.path.splitext(b)[0]
    cands = []
    if len(stem) >= 2 and stem[:2].upper() in ("GX", "GH"):
        cands.append("GL" + stem[2:])
    cands.append(stem)
    for s in cands:
        for ext in (".LRV", ".lrv"):
            p = os.path.join(d, s + ext)
            if os.path.exists(p):
                return p
    return None


def video_keyframes(path):
    """Presentation times (s) of the keyframes of the first video track, read
    from the MP4 index (no decoding). Returns (times, frame_duration)."""
    fsize = os.path.getsize(path)
    with open(path, "rb") as f:
        moov = None
        for typ, s, e in _iter_boxes(f, 0, fsize):
            if typ == b"moov":
                moov = (s, e)
        if moov is None:
            return None, None
        for typ, s, e in _iter_boxes(f, *moov):
            if typ != b"trak":
                continue
            info = {}

            def walk(s, e):
                for t2, cs, ce in _iter_boxes(f, s, e):
                    if t2 in CONTAINERS:
                        walk(cs, ce)
                    elif t2 in (b"stsd", b"stss", b"stts", b"ctts", b"elst", b"mdhd"):
                        info[t2] = _read(f, cs, ce)

            walk(s, e)
            stsd = info.get(b"stsd")
            if not stsd or len(stsd) < 16 or stsd[12:16] not in (b"avc1", b"avc3", b"hvc1", b"hev1", b"av01"):
                continue
            mdhd = info[b"mdhd"]
            ts = struct.unpack(">I", mdhd[20:24] if mdhd[0] == 1 else mdhd[12:16])[0]
            d = info[b"stts"]
            n = struct.unpack(">I", d[4:8])[0]
            durs = []
            for i in range(n):
                cnt, delta = struct.unpack(">II", d[8 + 8 * i: 16 + 8 * i])
                durs.append((cnt, delta))
            total = sum(c for c, _ in durs)
            dts = np.zeros(total + 1, dtype=np.int64)
            pos = 0
            for cnt, delta in durs:
                dts[pos + 1:pos + cnt + 1] = delta
                pos += cnt
            dts = np.cumsum(dts)[:-1]
            pts = dts.astype(np.float64)
            if b"ctts" in info:
                d = info[b"ctts"]
                n = struct.unpack(">I", d[4:8])[0]
                off = np.zeros(total, dtype=np.float64)
                pos = 0
                for i in range(n):
                    cnt, o = struct.unpack(">Ii", d[8 + 8 * i: 16 + 8 * i])
                    off[pos:pos + cnt] = o
                    pos += cnt
                pts = pts + off
            media_time = 0
            if b"elst" in info:
                d = info[b"elst"]
                ver = d[0]
                if struct.unpack(">I", d[4:8])[0] >= 1:
                    _, mt = struct.unpack(">Qq", d[8:24]) if ver == 1 else struct.unpack(">Ii", d[8:16])
                    if mt > 0:
                        media_time = mt
            if b"stss" in info:
                d = info[b"stss"]
                n = struct.unpack(">I", d[4:8])[0]
                idx = np.array(struct.unpack(f">{n}I", d[8:8 + 4 * n])) - 1
            else:
                idx = np.arange(total)
            times = (pts[idx] - media_time) / ts
            fdur = (durs[0][1] / ts) if durs else 1 / 30
            return np.sort(times[times >= 0]), fdur
    return None, None



def video_input_args(paths, tmpdir, start=0.0, durations=None):
    """ffmpeg input arguments that begin at `start` seconds on the combined
    timeline. Several files go through the concat demuxer with each file's
    duration declared (from the MP4 index), which lets ffmpeg seek the joined
    timeline exactly; a plain concat list without durations seeks badly."""
    start = max(0.0, start)
    ss = ["-ss", f"{start:.3f}"] if start > 0 else []
    if len(paths) == 1:
        return ss + ["-i", paths[0]]
    durations = durations or [mp4_info(p)[0] for p in paths]
    lst = os.path.join(tmpdir, "concat.txt")
    with open(lst, "w") as f:
        f.write("ffconcat version 1.0\n")
        for p, d in zip(paths, durations):
            f.write("file '" + os.path.abspath(p).replace("'", "'\\''") + "'\n")
            f.write(f"duration {d:.6f}\n")
    return ss + ["-f", "concat", "-safe", "0", "-i", lst]


HARMLESS_FFMPEG = ("Could not find ref with POC", "Error constructing the frame RPS", "First slice in a frame missing",
                   "Last message repeated", "mmco: unref short failure", "Missing reference picture")


def _relay_stderr(pipe):
    """Print ffmpeg's stderr, minus the decoder chatter that follows every seek
    into an open-GOP HEVC stream (those lead-in frames are discarded anyway)."""
    for raw in iter(pipe.readline, b""):
        line = raw.decode("utf8", "replace").rstrip()
        if line and not any(h in line for h in HARMLESS_FFMPEG):
            sys.stderr.write(line + "\n")
            sys.stderr.flush()


def print_laps(laps, out_csv=None, durations=None):
    if out_csv:
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            ns = laps.n_sectors if laps else 0
            w.writerow(["lap", "driver", "time_s", "time", "start_video_s", "start_video_mmss",
                        "start_chapter_time", "end_video_s", "is_best"] + [f"s{k + 1}" for k in range(ns)]
                       + ["peak_lat_g", "peak_brake_g", "peak_accel_g"])
            if laps and laps.times:
                best = int(np.argmin(laps.times))
                for i, lt in enumerate(laps.times):
                    w.writerow([i + 1, laps.driver_of_lap(i) or "", f"{lt:.3f}", fmt_lap(lt),
                                f"{laps.crossings[i]:.3f}", fmt_clock(laps.crossings[i]),
                                fmt_chapter(laps.crossings[i], durations),
                                f"{laps.crossings[i + 1]:.3f}", int(i == best)]
                               + [f"{v:.3f}" if v is not None else "" for v in laps.sector_times[i]]
                               + [f"{v:.2f}" for v in (lambda p: (max(p[0], p[1]), p[3], p[2]))(
                                   laps.tel.g_peaks(laps.crossings[i], laps.crossings[i + 1]))])
    if not laps or not laps.times:
        print(laps.explain() if laps else "No telemetry.")
        print("  Tip: run with --laps-only and no --sf-time to let the tool find the lap itself, then\n"
              "  give --sf-time a moment when the car is crossing the real start/finish line.")
        return
    best = int(np.argmin(laps.times))
    med = float(np.median(laps.times))
    has_drv = bool(laps.drivers)
    multi = durations is not None and len(durations) > 1
    ns = laps.n_sectors if laps.n_sectors >= 2 else 0
    dcol = f"  {'Driver':<10}" if has_drv else ""
    ccol = "  file@time" if multi else ""
    scol = "".join(f"  {'S' + str(k + 1):>6}" for k in range(ns))
    tel = laps.tel
    peaks = [tel.g_peaks(laps.crossings[i], laps.crossings[i + 1]) for i in range(len(laps.times))]
    print(f"\n{'Lap':>4}{dcol}  {'Time':>9}  {'Gap':>7}{scol}  {'LatG':>5} {'BrkG':>5}   starts at{ccol}")
    for i, lt in enumerate(laps.times):
        gap = lt - laps.times[best]
        mark = "  <- BEST" if i == best else ("  (long)" if lt > 1.6 * med else "")
        drv = f"  {(laps.driver_of_lap(i) or '')[:10]:<10}" if has_drv else ""
        ch = f"  {fmt_chapter(laps.crossings[i], durations)}" if multi else ""
        sc = "".join(f"  {fmt_sector(v):>6}" for v in laps.sector_times[i]) if ns else ""
        pk = f"  {max(peaks[i][0], peaks[i][1]):5.2f} {peaks[i][3]:5.2f}"
        print(f"{i + 1:>4}{drv}  {fmt_lap(lt):>9}  {('+%.2f' % gap) if i != best else '':>7}{sc}{pk}   "
              f"{fmt_clock(laps.crossings[i]):>9}{ch}{mark}")
    who = f" by {laps.driver_of_lap(best)}" if has_drv else ""
    print(f"\nBest lap: {fmt_lap(laps.times[best])} (lap {best + 1}{who}), {len(laps.times)} timed laps")
    if ns:
        bs = laps.best_sectors(len(laps.times))
        if all(v is not None for v in bs):
            print("Best sectors: " + "  ".join(f"S{k + 1} {fmt_sector(v)}" for k, v in enumerate(bs))
                  + f"  ->  theoretical best {fmt_lap(sum(bs))}")
    if has_drv:
        for name in dict.fromkeys(n for _, n in laps.drivers):
            idx = [i for i in range(len(laps.times)) if laps.driver_of_lap(i) == name]
            if idx:
                mine = [laps.times[i] for i in idx]
                miles = sum(laps.lap_dist[i] for i in idx) / 1609.344
                plat = max(max(peaks[i][0], peaks[i][1]) for i in idx)
                pbrk = max(peaks[i][3] for i in idx)
                print(f"  {name}: {len(mine)} laps, best {fmt_lap(min(mine))}, {miles:.1f} mi, "
                      f"peak {plat:.2f} g lateral / {pbrk:.2f} g braking")
    if tel.g_source:
        print(f"  (g figures from the {'accelerometer' if tel.g_source == 'accel' else 'GPS'})")
    print()



_LUT_Y = [min(255, max(0, 16 + round(v * 219 / 255))) for v in range(256)]
_LUT_C = [min(255, max(0, 128 + round((v - 128) * 224 / 255))) for v in range(256)]


def frame_nbytes(W, H, fmt):
    return W * H * 4 if fmt == "rgba" else W * H + 2 * (W // 2) * (H // 2) + W * H


def encode_frame(img, fmt):
    """RGBA PIL image -> raw bytes for ffmpeg. 'yuva420p' is ~40% smaller and
    saves ffmpeg a per-frame colour conversion (limited-range BT.601, which is
    what ffmpeg would have produced itself)."""
    if fmt == "rgba":
        return img.tobytes()
    y = img.convert("L").point(_LUT_Y)
    _, cb, cr = img.reduce(2).convert("YCbCr").split()
    return y.tobytes() + cb.point(_LUT_C).tobytes() + cr.point(_LUT_C).tobytes() + img.getchannel("A").tobytes()


# ----------------------------------------------------------------------------
# Parallel rendering (worker processes draw frames into shared memory)
# ----------------------------------------------------------------------------

_WORKER = None


def _worker_init(shm_name, slot_bytes, tel, laps, rkw, font_path, fmt):
    global _WORKER
    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(name=shm_name)
    rkw = dict(rkw)
    rkw["fonts"] = Fonts(font_path)
    _WORKER = (shm, slot_bytes, Renderer(tel, laps, **rkw), fmt)


def _worker_render(task):
    i, slot, T = task
    shm, sb, r, fmt = _WORKER
    shm.buf[slot * sb:(slot + 1) * sb] = encode_frame(r.frame(T), fmt)
    return i, slot


def render_frames(renderer_kw, tel, laps, font_path, W, H, times, jobs, sink, progress, fmt="yuva420p"):
    """Render every frame time in order and pass its raw bytes to sink(data);
    calls progress(i) after each. Uses `jobs` worker processes (inline if <= 1)."""
    if jobs <= 1:
        r = Renderer(tel, laps, **dict(renderer_kw, fonts=Fonts(font_path)))
        for i, T in enumerate(times):
            sink(encode_frame(r.frame(T), fmt))
            progress(i)
        return
    from multiprocessing import shared_memory
    slot_bytes = frame_nbytes(W, H, fmt)
    nslots = jobs * 3
    shm = shared_memory.SharedMemory(create=True, size=slot_bytes * nslots)
    free = queue.Queue()
    for k in range(nslots):
        free.put(k)

    def tasks():
        for i, T in enumerate(times):
            yield i, free.get(), T

    ctx = mp.get_context(os.environ.get("TELEMETRY_OVERLAY_START_METHOD") or None)
    pool = ctx.Pool(jobs, initializer=_worker_init,
                    initargs=(shm.name, slot_bytes, tel, laps, renderer_kw, font_path, fmt))
    try:
        for i, slot in pool.imap(_worker_render, tasks(), chunksize=1):
            view = shm.buf[slot * slot_bytes:(slot + 1) * slot_bytes]
            try:
                sink(view)
            finally:
                view.release()
            free.put(slot)
            progress(i)
    finally:
        pool.terminate()
        pool.join()
        shm.close()
        try:
            shm.unlink()
        except FileNotFoundError:
            pass


def ffmpeg_has_filters(*names):
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    except OSError:
        return False
    return all(f" {n} " in out for n in names)


def gpu_filter_graph(out_w, out_h, fmt):
    # generic hwupload accepts yuva420p (hwupload_cuda does not)
    ov = "[1:v]hwupload[ov]" if fmt == "yuva420p" else "[1:v]format=yuva420p,hwupload[ov]"
    return (f"[0:v]scale_cuda={out_w}:{out_h}:format=yuv420p[bg];{ov};"
            f"[bg][ov]overlay_cuda=0:0:shortest=1[out]")


def gpu_graph_works(inp_args, ov_w, ov_h, out_w, out_h, vcodec, fmt="yuva420p", ctags=None):
    """Dry-run the all-GPU filter graph on the first fraction of a second."""
    filt = gpu_filter_graph(out_w, out_h, fmt)
    ctag = color_args(ctags) if ctags else []
    cmd = (["ffmpeg", "-v", "warning", "-nostats", "-y", "-init_hw_device", "cuda=cu", "-filter_hw_device", "cu",
            "-hwaccel", "cuda", "-hwaccel_device", "cu", "-hwaccel_output_format", "cuda", "-t", "0.3"]
           + inp_args + ["-f", "rawvideo", "-pix_fmt", fmt, "-s", f"{ov_w}x{ov_h}", "-r", "30"] + ctag + ["-i", "-",
                         "-filter_complex", filt, "-map", "[out]", "-frames:v", "2", "-c:v", vcodec, "-f", "null", "-"])
    try:
        p = subprocess.run(cmd, input=bytes(frame_nbytes(ov_w, ov_h, fmt) * 3), capture_output=True, timeout=60)
        return p.returncode == 0, p.stderr.decode("utf8", "replace").strip()
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, str(e)




class OverlayFeed:
    """Carries the overlay frames into ffmpeg. 'tcp' uses a local socket (much
    faster than a pipe on Windows); 'pipe' uses ffmpeg's stdin."""

    def __init__(self, kind="tcp"):
        self.kind = kind
        self.sock = self.conn = self.proc = None
        if kind == "tcp":
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.bind(("127.0.0.1", 0))
                self.sock.listen(1)
                self.url = f"tcp://127.0.0.1:{self.sock.getsockname()[1]}"
            except OSError:
                self.sock = None
                self.kind = "pipe"
        if self.kind == "pipe":
            self.url = "-"

    def start(self, cmd):
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE if self.kind == "pipe" else None,
                                     stderr=subprocess.PIPE)
        threading.Thread(target=_relay_stderr, args=(self.proc.stderr,), daemon=True).start()
        if self.kind == "tcp":
            self.sock.settimeout(0.5)
            deadline = time.time() + 180
            while True:
                try:
                    self.conn, _ = self.sock.accept()
                    break
                except socket.timeout:
                    if self.proc.poll() is not None:
                        raise SystemExit(f"ffmpeg exited before reading the overlay (exit code {self.proc.returncode})")
                    if time.time() > deadline:
                        self.proc.kill()
                        raise SystemExit("ffmpeg never connected to the overlay feed")
            self.conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 << 20)
            self.write = self.conn.sendall
        else:
            self.write = self.proc.stdin.write
        return self.write

    def finish(self):
        """Close the feed and wait for ffmpeg; returns its exit code."""
        for fn in ((lambda: self.conn.shutdown(socket.SHUT_WR)) if self.conn is not None else None,
                   self.conn.close if self.conn is not None else None,
                   self.sock.close if self.sock is not None else None,
                   self.proc.stdin.close if self.proc.stdin is not None else None):
            if fn is not None:
                try:
                    fn()
                except OSError:
                    pass
        return self.proc.wait()


def feed_frames(cmd_template, kind, source):
    """Run ffmpeg with the overlay input placeholder replaced by the feed's url,
    push frames from source(write) and return ffmpeg's exit code."""
    feed = OverlayFeed(kind)
    cmd = [feed.url if a == "OVERLAY_INPUT" else a for a in cmd_template]
    write = feed.start(cmd)
    try:
        source(write)
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
        pass
    return feed.finish()


def run_bench(cmd, filt_bg, renderer_kw, tel, laps, font_path, W, H, start, ofps, jobs, dur, feed_kind="tcp",
              fmt="yuva420p"):
    """Time each stage separately so the bottleneck is obvious."""
    print("\n=== Benchmark ===")
    try:
        ver = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
        print(ver)
    except Exception:
        pass
    print(f"CPU threads: {os.cpu_count()}, workers: {jobs}, overlay {W}x{H} as {fmt} via {feed_kind}, "
          f"telemetry points: {len(tel.t)} GPS / {len(tel.g_t)} g ({tel.g_source})")

    # 1) per-element cost in one process
    r = Renderer(tel, laps, **dict(renderer_kw, fonts=Fonts(font_path)))
    T = start + min(dur, 30.0)
    st = laps.state(T, r.hold) if laps else {"lap": None}
    img = r.static.copy()
    d = ImageDraw.Draw(img)
    x, y, sp, dist = tel.at(T)

    def tm(fn, n=20):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n * 1000

    parts = [("copy static", lambda: r.static.copy()),
             (f"to {fmt}", lambda: encode_frame(img, fmt)),
             ("lap state", lambda: laps.state(T, r.hold) if laps else None)]
    if r.dial_c:
        parts.append(("speedo", lambda: r._draw_speedo(img, d, sp * r.conv, 100)))
    if r.g_c or r.brake_box:
        parts.append(("g-meter", lambda: r._draw_gmeter(img, d, T, True, st)))
    if r.map_box:
        parts.append(("map dot", lambda: img.alpha_composite(r.dot, (100, 100))))
    if r.elev_box:
        parts.append(("elevation", lambda: r._draw_elev(img, d, st, dist)))
    if r.trace_box:
        parts.append(("speed trace", lambda: r._draw_trace(img, d, st, T)))
    if r.info_box:
        parts.append(("clock strip", lambda: r._draw_info(img, d, T, dist)))
    if r.panel:
        parts.append(("lap panel", lambda: r._draw_laps(img, d, T, st)))
    if r.sector_box:
        parts.append(("sectors", lambda: r._draw_sectors(img, d, st)))
    parts.append(("lap list", lambda: r._draw_list(img, d, st)))
    print("\nPer-frame drawing cost (single process):")
    tot = 0.0
    for name, fn in parts:
        ms = tm(fn)
        tot += ms
        print(f"  {name:<14}{ms:6.1f} ms")
    full = tm(lambda: encode_frame(r.frame(T), fmt), 10)
    print(f"  {'whole frame':<14}{full:6.1f} ms  -> {1000 / full:.0f} fps from one process")

    # 2) parallel drawing throughput (no ffmpeg), steady state after startup
    n = 400
    warm = 60
    times = [start + i / ofps for i in range(n)]
    for j in sorted({1, jobs}):
        marks = {}

        def prog(i, marks=marks):
            if i == warm or i == n - 1:
                marks[i] = time.perf_counter()

        t0 = time.perf_counter()
        render_frames(renderer_kw, tel, laps, font_path, W, H, times, j, lambda data: None, prog, fmt)
        total = time.perf_counter() - t0
        steady = (n - 1 - warm) / max(marks[n - 1] - marks[warm], 1e-6)
        print(f"Drawing with {j} worker(s): {steady:.0f} fps steady ({steady / ofps:.1f}x realtime), "
              f"startup {total - (n / steady):.1f}s")

    # 3) ffmpeg alone: feed it the same frame over and over (20 s of video)
    frame = encode_frame(r.frame(T), fmt)
    base_cmd = list(cmd)
    base_cmd[base_cmd.index("-t") + 1] = "20"
    base_cmd = base_cmd[:-1] + ["-f", "null", "-"]
    for k in ("-c:a", "-b:a", "-movflags"):
        if k in base_cmd:
            i = base_cmd.index(k)
            del base_cmd[i:i + 2]
    nf = int(20 * ofps) + 1

    def run(label, c, kind, with_overlay=True):
        print(f"{label}: ", end="", flush=True)
        t0 = time.perf_counter()
        if with_overlay:
            def src(write):
                for _ in range(nf):
                    write(frame)
            rc = feed_frames(c, kind, src)
        else:
            rc = subprocess.run(c).returncode
        el = time.perf_counter() - t0
        print(f"{nf / el:.0f} fps ({nf / el / ofps:.1f}x realtime)" if rc == 0 else f"failed (exit {rc})")

    run(f"FFmpeg alone, overlay via {feed_kind} (decode + scale + composite + encode)", base_cmd, feed_kind)
    other = "pipe" if feed_kind == "tcp" else "tcp"
    run(f"FFmpeg alone, overlay via {other}", base_cmd, other)
    # no overlay input at all: decode + scale + encode only
    c = list(base_cmd)
    i = c.index("-f", c.index("-i") + 1)          # the rawvideo input block
    j = c.index("OVERLAY_INPUT") + 1
    del c[i:j]
    c[c.index("-filter_complex") + 1] = filt_bg
    run("FFmpeg alone, no overlay at all (decode + scale + encode)", c, feed_kind, with_overlay=False)
    print("\nThe slowest of these is your bottleneck. If FFmpeg alone is slow, more workers won't help;\n"
          "try --gpu-filters (needs an FFmpeg build with scale_cuda/overlay_cuda) or --height 1080.\n"
          "If drawing is slow, try --overlay-fps 15 or --hide some elements.\n")

# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Render a racing telemetry overlay onto GoPro video using its GPS and accelerometer.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("Typical use:")[1])
    ap.add_argument("videos", nargs="+", help="GoPro MP4(s). Pass chapters in order (GX01xxxx, GX02xxxx, ...)")
    ap.add_argument("-o", "--output", help="output video (default: <first input>_overlay.mp4)")

    g = ap.add_argument_group("start/finish line (pick one; default = auto)")
    g.add_argument("--sf-time", help="video time when the car is ON the start/finish line. Formats: 1155, 19:15, "
                   "or chapter@time such as 2@0:45 (chapter 2 = second file given, 45 s in)")
    g.add_argument("--sf-point", help="start/finish location as 'lat,lon' (e.g. from Google Maps)")
    g.add_argument("--sf-width", type=float, default=30.0, help="width of the timing line in meters (default 30)")
    g.add_argument("--min-lap", type=float, default=20.0, help="ignore crossings closer than this many seconds")
    g.add_argument("--green", help="race start: ignore line crossings before this video time. "
                                   "Laps are counted from the first crossing after it; before that shows PACE LAP")
    g.add_argument("--checkered", help="time you take the checkered flag; the cool-down lap after it isn't "
                                       "timed and the overlay shows the final result")
    g.add_argument("--sectors", type=int, default=3, help="number of sectors per lap (default 3, 1 = off)")
    g.add_argument("--driver", action="append", default=[], metavar="NAME[@WHEN]",
                   help="driver name and when they took over. Repeat for each stint, e.g. "
                        "--driver Chris --driver Dave@lap38 --driver Chris@3@12:40 (chapter 3, 12:40 in). "
                        "The first driver needs no @when")

    g = ap.add_argument_group("modes")
    g.add_argument("--laps-only", action="store_true", help="just print laps and save a map PNG, no video")
    g.add_argument("--frame", help="render a single still PNG at this video time (same formats as --sf-time)")
    g.add_argument("--start", default="0", help="start rendering at this time (same formats as --sf-time)")
    g.add_argument("--duration", help="only render this long (seconds or mm:ss)")
    g.add_argument("--use-lrv", action="store_true",
                   help="use the low-res .LRV proxy as the video (fast drafts); GPS still comes from the MP4")

    g = ap.add_argument_group("appearance")
    g.add_argument("--kmh", action="store_true", help="show km/h instead of MPH")
    g.add_argument("--speed-max", type=float, help="speedo full-scale value (default: auto)")
    g.add_argument("--g-max", type=float, default=1.5, help="G-meter full scale in g (default 1.5)")
    g.add_argument("--brake-g", type=float, default=0.5,
                   help="deceleration in g that lights the BRAKE lamp (default 0.5; 0.3 catches engine braking too)")
    g.add_argument("--g-peaks", choices=["lap", "session", "off"], default="lap",
                   help="peak markers on the G-meter: reset each lap (default), whole session, or off")
    g.add_argument("--scale", type=float, default=1.0, help="size of all overlay elements (0.8 = smaller)")
    g.add_argument("--hide", default="", help="comma-separated elements to leave out: " + ",".join(ALL_ELEMENTS))
    g.add_argument("--hold", type=float, default=3.0, help="seconds to hold a finished lap time (default 3)")
    g.add_argument("--banner", type=float, default=5.0, help="seconds to show the NEW BEST LAP banner (default 5)")
    g.add_argument("--lap-list", type=int, default=6, help="how many recent laps to list (0 = hide)")
    g.add_argument("--team", help="team name shown in a panel across the top, e.g. \"Bill's Discount Garage\"")
    g.add_argument("--car", help="car description shown under the team name, e.g. \"2007 BMW 328i\"")
    g.add_argument("--number", help="car number, shown as a badge next to the car, e.g. 88")
    g.add_argument("--logo", help="team logo image (PNG with transparency works best) shown across the top")
    g.add_argument("--logo-height", type=float, default=80, help="logo height in px at 1080p (default 80)")
    g.add_argument("--sponsor", action="append", default=[], metavar="IMAGE",
                   help="sponsor logo image for a slot along the bottom (repeat for more slots)")
    g.add_argument("--sponsor-slots", type=int, default=0,
                   help="number of sponsor slots to show; empty ones are drawn as placeholders")
    g.add_argument("--tz", type=float, help="time zone offset in hours for the clock (default: this PC's zone)")
    g.add_argument("--font", help="path to a .ttf font")

    g = ap.add_argument_group("output / encoding")
    g.add_argument("--height", type=int, help="output height, e.g. 1080 or 2160 (default: same as source)")
    g.add_argument("--overlay-fps", type=float, default=30.0, help="overlay update rate (default 30)")
    g.add_argument("--vcodec", default="libx264",
                   help="libx264 (default), h264_videotoolbox (Mac), h264_nvenc (NVIDIA), hevc_nvenc ...")
    g.add_argument("--nvidia", action="store_true",
                   help="use the NVIDIA card: GPU decoding (-hwaccel cuda) + NVENC encoding. Much faster.")
    g.add_argument("--hwaccel", help="ffmpeg hardware decoder: cuda (NVIDIA), qsv (Intel), d3d11va, auto")
    g.add_argument("--cq", type=int, default=19, help="NVENC quality, lower = better (default 19)")
    g.add_argument("--bench", action="store_true",
                   help="measure each stage of the pipeline (drawing, workers, ffmpeg) and report the bottleneck")
    g.add_argument("--overlay-format", choices=["yuva420p", "rgba"], default="yuva420p",
                   help="pixel format the overlay is sent to ffmpeg in (yuva420p = smaller and faster)")
    g.add_argument("--feed", choices=["tcp", "pipe"], default="tcp",
                   help="how overlay frames reach ffmpeg: local socket (default, fast on Windows) or stdin pipe")
    g.add_argument("--jobs", type=int, help="worker processes drawing the overlay (default: about half your CPU threads)")
    g.add_argument("--gpu-filters", choices=["auto", "on", "off"], default="auto",
                   help="with --nvidia, also scale and composite on the GPU (scale_cuda/overlay_cuda). "
                        "auto = use it if your FFmpeg build supports it (default)")
    g.add_argument("--crf", type=int, default=18, help="x264 quality, lower = better (default 18)")
    g.add_argument("--preset", default="medium", help="x264 preset (default medium)")

    g = ap.add_argument_group("telemetry")
    g.add_argument("--gps-offset", type=float, default=0.0, help="shift GPS in time by this many seconds")
    g.add_argument("--telemetry-csv", help="use this CSV (t,lat,lon[,speed m/s][,alt m]) instead of the GoPro GPS")
    g.add_argument("--dump-telemetry", help="write the extracted GPS to this CSV")
    args = ap.parse_args()
    if not args.laps_only:
        ensure_prerequisites(need_ffmpeg=True)

    if args.nvidia:
        args.hwaccel = args.hwaccel or "cuda"
        if args.vcodec == "libx264":
            args.vcodec = "h264_nvenc"
    hw = ["-hwaccel", args.hwaccel] if args.hwaccel else []
    hide = {h.strip().lower() for h in args.hide.split(",") if h.strip()}
    bad = hide - set(ALL_ELEMENTS)
    if bad:
        raise SystemExit(f"--hide: unknown element(s) {', '.join(sorted(bad))}. Choose from: {', '.join(ALL_ELEMENTS)}")
    if args.lap_list == 0:
        hide.add("list")

    for p in args.videos + ([args.logo] if args.logo else []) + list(args.sponsor):
        if not os.path.exists(p):
            raise SystemExit(f"File not found: {p}")
        if p.lower().endswith(".lrv"):
            raise SystemExit("Pass the .MP4 files, not the .LRV (the LRV is just a low-res preview copy).")

    # --- telemetry
    if args.telemetry_csv:
        raw = load_csv_telemetry(args.telemetry_csv)
        durations = [mp4_info(p)[0] for p in args.videos]
    else:
        raw, durations = extract_gopro_gps(args.videos)
    if args.dump_telemetry:
        with open(args.dump_telemetry, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "lat", "lon", "alt", "speed"])
            for i in range(len(raw["t"])):
                w.writerow([f"{raw['t'][i]:.3f}", f"{raw['lat'][i]:.7f}", f"{raw['lon'][i]:.7f}",
                            f"{raw['alt'][i]:.1f}" if raw.get("alt") is not None else "",
                            f"{raw['speed'][i]:.3f}" if raw["speed"] is not None else ""])
        print(f"Wrote {args.dump_telemetry}")
    tel = Telemetry(raw, args.gps_offset)
    total_dur = sum(durations)
    args.sf_time = parse_time(args.sf_time, durations)
    args.green = parse_time(args.green, durations)
    args.checkered = parse_time(args.checkered, durations)
    args.frame = parse_time(args.frame, durations)
    args.start = parse_time(args.start, durations)
    args.duration = parse_time(args.duration)
    if len(durations) > 1:
        print("Chapters: " + ", ".join(f"{i + 1}={os.path.basename(p)} starts at {fmt_clock(sum(durations[:i]))}"
                                       for i, p in enumerate(args.videos)))

    # --- laps
    kw = dict(width=args.sf_width, min_lap=args.min_lap, green=args.green, checkered=args.checkered,
              n_sectors=args.sectors)
    if args.sf_time is not None:
        laps = Laps.from_time(tel, args.sf_time, **kw)
    elif args.sf_point:
        lat, lon = (float(v) for v in args.sf_point.split(","))
        laps = Laps.from_point(tel, lat, lon, **kw)
    else:
        laps = Laps.auto(tel, **kw)
        print("No start/finish given: timing line placed at the top-speed point "
              "(lap times are still valid; use --sf-time or --sf-point for the real line).")

    base = os.path.splitext(args.output or args.videos[0])[0]
    if not args.output:
        base += "_overlay"
    if args.driver:
        laps.set_drivers(args.driver, durations)
    print_laps(laps, base + "_laps.csv", durations)

    # --- video
    src_videos = args.videos
    if args.use_lrv:
        lrvs = [lrv_for(p) for p in args.videos]
        if all(lrvs):
            src_videos = lrvs
            print("Using LRV proxies for video.")
        else:
            print("WARNING: .LRV files not found next to the MP4s; using the MP4s.", file=sys.stderr)

    sw, sh, fps = probe(src_videos[0])
    ctags = probe_color(src_videos[0])
    out_h = args.height or sh
    out_h -= out_h % 2
    out_w = int(round(sw * out_h / sh / 2)) * 2
    ov_h = min(out_h, 1080) if out_h >= 720 else out_h
    ov_w = int(round(out_w * ov_h / out_h / 2)) * 2
    units = "kmh" if args.kmh else "mph"
    fonts = Fonts(args.font)
    tz = _dt.timezone(_dt.timedelta(hours=args.tz)) if args.tz is not None else None

    def make_renderer():
        return Renderer(tel, laps, ov_w, ov_h, units, args.speed_max, fonts, args.hold, args.banner,
                        args.lap_list, hide, args.scale, args.g_max, tz, args.green, args.g_peaks,
                        args.logo, args.sponsor, args.sponsor_slots, args.logo_height, args.team, args.car, args.number,
                        args.brake_g)

    if args.laps_only:
        r = make_renderer()
        if r.map_box:
            mx0, my0, mx1, my1 = r.map_box
            r.static.crop((mx0, my0, mx1, my1)).save(base + "_map.png")
            print(f"Wrote {base}_map.png and {base}_laps.csv")
        else:
            print(f"Wrote {base}_laps.csv")
        return

    renderer = make_renderer()

    if args.frame is not None:
        need("ffmpeg")
        with tempfile.TemporaryDirectory() as td:
            png = os.path.join(td, "f.png")
            inp = video_input_args(src_videos, td, args.frame, durations)
            subprocess.run(["ffmpeg", "-v", "error", "-y"] + hw + inp +
                           ["-frames:v", "1", "-vf", f"scale={out_w}:{out_h}", png], check=True)
            bg = Image.open(png).convert("RGBA")
        ov = renderer.frame(args.frame)
        if ov.size != bg.size:
            ov = ov.resize(bg.size, Image.LANCZOS)
        bg.alpha_composite(ov)
        outp = f"{base}_frame_{args.frame:g}s.png"
        bg.convert("RGB").save(outp)
        print(f"Wrote {outp}")
        return

    need("ffmpeg")
    start = max(0.0, args.start)
    dur = (args.duration if args.duration else total_dur - start)
    dur = min(dur, total_dur - start)
    ofps = min(args.overlay_fps, fps)
    nframes = int(math.ceil(dur * ofps)) + 1
    out_path = args.output or (base + ".mp4")

    jobs = args.jobs if args.jobs else max(1, min(8, (os.cpu_count() or 2) // 2))
    renderer_kw = dict(W=ov_w, H=ov_h, units=units, speed_max=args.speed_max, hold=args.hold, banner=args.banner,
                       show_list=args.lap_list, hide=hide, scale=args.scale, g_max=args.g_max, tz=tz,
                       race_start=args.green, g_peaks=args.g_peaks, logo=args.logo, sponsors=args.sponsor,
                       sponsor_slots=args.sponsor_slots, logo_height=args.logo_height, team=args.team, car=args.car,
                       number=args.number, brake_g=args.brake_g)

    with tempfile.TemporaryDirectory() as td:
        inp = video_input_args(src_videos, td, start, durations)
        if args.vcodec == "h264_nvenc" and out_w > 4096:
            print("Output wider than 4096 px: H.264 NVENC can't do that, switching to hevc_nvenc.")
            args.vcodec = "hevc_nvenc"

        # all-GPU filter graph?
        use_gpu_filters = False
        if args.hwaccel == "cuda" and args.vcodec.endswith("_nvenc") and args.gpu_filters != "off" \
                and (ov_w, ov_h) == (out_w, out_h):
            if not ffmpeg_has_filters("scale_cuda", "overlay_cuda"):
                if args.gpu_filters == "on":
                    print("This FFmpeg build has no scale_cuda/overlay_cuda filters; using CPU filters.")
            else:
                ok, err = gpu_graph_works(inp, ov_w, ov_h, out_w, out_h, args.vcodec, args.overlay_format, ctags)
                if ok:
                    use_gpu_filters = True
                else:
                    print("GPU filter graph failed its self-test; using CPU filters (decode/encode still on the GPU).")
                    if args.gpu_filters == "on" or args.bench:
                        lines = [ln for ln in err.splitlines() if ln.strip()] or ["(no message)"]
                        print("  ffmpeg said:")
                        for ln in lines[-8:]:
                            print("    " + ln)

        if use_gpu_filters:
            filt = gpu_filter_graph(out_w, out_h, args.overlay_format)
            head = ["-init_hw_device", "cuda=cu", "-filter_hw_device", "cu", "-hwaccel", "cuda",
                    "-hwaccel_device", "cu", "-hwaccel_output_format", "cuda"]
        else:
            scale_bg = f"scale={out_w}:{out_h}," if (out_w, out_h) != (sw, sh) else ""
            scale_ov = f"scale={out_w}:{out_h}:flags=bicubic," if (ov_w, ov_h) != (out_w, out_h) else ""
            ovf = f"[1:v]{scale_ov}format=yuva420p[ov]" if scale_ov or args.overlay_format == "rgba" else "[1:v]null[ov]"
            filt = (f"[0:v]{scale_bg}setsar=1[bg];{ovf};"
                    f"[bg][ov]overlay=0:0:shortest=1:format=yuv420,format=yuv420p[out]")
            head = hw
        cmd = ["ffmpeg", "-v", "error", "-nostats", "-y"] + head + inp
        cmd += ["-f", "rawvideo", "-pix_fmt", args.overlay_format, "-s", f"{ov_w}x{ov_h}", "-r", f"{ofps:g}"]
        cmd += color_args(ctags) + ["-i", "OVERLAY_INPUT",
                                    "-filter_complex", filt, "-map", "[out]", "-map", "0:a:0?", "-t", f"{dur:.3f}",
                                    "-c:v", args.vcodec] + color_args(ctags)
        if args.vcodec == "libx264":
            cmd += ["-crf", str(args.crf), "-preset", args.preset]
        elif args.vcodec.endswith("_nvenc"):
            cmd += ["-preset", "p5", "-rc", "vbr", "-cq", str(args.cq), "-b:v", "0",
                    "-maxrate", "120M" if out_h > 1440 else "60M", "-bufsize", "200M", "-spatial-aq", "1"]
        elif args.vcodec.endswith("_videotoolbox") or args.vcodec.endswith("_qsv"):
            cmd += ["-b:v", "40M" if out_h > 1440 else "20M"]
        cmd += ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out_path]

        if use_gpu_filters:
            filt_bg = f"[0:v]scale_cuda={out_w}:{out_h}:format=yuv420p[out]"
        else:
            filt_bg = f"[0:v]{scale_bg}setsar=1,format=yuv420p[out]"
        if args.bench:
            run_bench(cmd, filt_bg, renderer_kw, tel, laps, args.font, ov_w, ov_h, start, ofps, jobs, dur, args.feed,
                      args.overlay_format)
            return

        print(f"Rendering {fmt_hms(dur)} -> {out_path} ({out_w}x{out_h}, overlay {ov_w}x{ov_h} @ {ofps:g} fps, "
              f"{jobs} worker{'s' if jobs != 1 else ''}"
              + (", GPU filters" if use_gpu_filters else "") + ")")
        t0 = time.time()
        last_report = 0.0
        times = [start + i / ofps for i in range(nframes)]

        def progress(i):
            nonlocal last_report
            now = time.time()
            if now - last_report >= 1.0 or i == nframes - 1:
                last_report = now
                done = (i + 1) / nframes
                elapsed = now - t0
                eta = elapsed / done - elapsed
                rate = (i / ofps) / max(elapsed, 1e-3)
                sys.stdout.write(f"\r  {done * 100:5.1f}%  |  {fmt_hms(i / ofps)} of {fmt_hms(dur)} rendered  |  "
                                 f"elapsed {fmt_hms(elapsed)}  |  ETA {fmt_hms(eta)}  |  {rate:.2f}x realtime   ")
                sys.stdout.flush()

        rc = feed_frames(cmd, args.feed, lambda write: render_frames(
            renderer_kw, tel, laps, args.font, ov_w, ov_h, times, jobs, write, progress, args.overlay_format))
        if rc != 0:
            hint = ""
            if args.hwaccel or not args.vcodec.startswith("libx264"):
                hint = ("\nHardware mode failed. Update your NVIDIA driver and use a recent FFmpeg build "
                        "(e.g. from gyan.dev), or retry without --nvidia/--hwaccel.")
            raise SystemExit(f"ffmpeg failed (exit code {rc}){hint}")
        total = time.time() - t0
        print(f"\nDone in {fmt_hms(total)} ({dur / max(total, 1e-3):.2f}x realtime) -> {out_path}")

if __name__ == "__main__":
    mp.freeze_support()
    main()
