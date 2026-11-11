"""Local web UI server for Bad CA!! (stdlib only). Reuses ca.pipeline for probe/prep/train/export; never computes
CA frames itself (the browser does, from the export, with player/nca_core.mjs).

    .venv/bin/python ui/server.py [--host 127.0.0.1] [--port 8765]
"""
import argparse, base64, hashlib, json, os, re, secrets, signal, subprocess, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
from ca import pipeline  # noqa: E402

STATIC = os.path.join(ROOT, "ui", "static")
PLAYER = os.path.join(ROOT, "player")
RUNS = os.path.join(ROOT, "runs", "pipe")
UIDIR = os.path.join(ROOT, "runs", "ui")
UPLOADS = os.path.join(UIDIR, "uploads")
IMPORTED = os.path.join(UIDIR, "imported")
CACHE = os.path.join(ROOT, "cache", "pipe")
MAX_UPLOAD = 200 << 20
MAX_EXPORT = 5 << 20
BUILTIN = {"badapple": os.path.join(ROOT, "refs", "Bad-Apple-Starter", "BadApple!!.mp4"),
           "color_disc": os.path.join(ROOT, "tests", "data", "color_disc_96x64.mp4")}
RUN_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,79}$")
IMP_RE = re.compile(r"^imp-([0-9a-f]{16})$")
UP_RE = re.compile(r"^up-([0-9a-f]{16})(\.[a-z0-9]{1,5})$")
EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,5}$")
C12 = os.path.join(ROOT, "runs", "c12")   # read-only: phased NCA (ncab-v2) and explicit-rule (dlrule-v1) exports
C12_RE = re.compile(r"^c12~[A-Za-z0-9_.-]+(~[A-Za-z0-9_.-]+)*$")
JOBS = {}  # run name -> Popen (children of this server, so we can reap them)
MIME = {".html": "text/html; charset=utf-8", ".css": "text/css", ".js": "text/javascript", ".mjs": "text/javascript",
        ".woff2": "font/woff2", ".txt": "text/plain; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png"}


class ApiError(Exception):
    def __init__(self, code, msg): super().__init__(msg); self.code = code


def under(root, *parts):
    """Join and resolve; refuse anything that escapes root."""
    p = os.path.realpath(os.path.join(root, *parts))
    if p != os.path.realpath(root) and not p.startswith(os.path.realpath(root) + os.sep): raise ApiError(400, "path outside allowed root")
    return p


def rjson(path, default=None):
    try:
        with open(path) as f: return json.load(f)
    except (OSError, ValueError): return default


def wjson(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(obj, f, indent=1)
    os.replace(tmp, path)


# ---------- videos ----------
def video_path(vid):
    if vid in BUILTIN: return BUILTIN[vid]
    m = UP_RE.match(str(vid))
    if not m: raise ApiError(400, "bad video id")
    p = under(UPLOADS, m.group(1) + m.group(2))
    if not os.path.isfile(p): raise ApiError(404, "unknown video")
    return p


def list_videos():
    out = [dict(id=k, name=os.path.relpath(p, ROOT), builtin=True) for k, p in BUILTIN.items() if os.path.isfile(p)]
    if os.path.isdir(UPLOADS):
        for f in sorted(os.listdir(UPLOADS)):
            if f.endswith(".json"): continue
            vid = "up-" + f
            if UP_RE.match(vid):
                side = rjson(os.path.join(UPLOADS, f + ".json"), {})
                out.append(dict(id=vid, name=side.get("name", f), builtin=False, probe=side.get("probe")))
    return out


def upload(h):
    n = int(h.headers.get("Content-Length") or -1)
    if n <= 0: raise ApiError(411, "Content-Length required (raw body upload)")
    if n > MAX_UPLOAD: raise ApiError(413, f"file larger than {MAX_UPLOAD >> 20} MB")
    disp = (h.headers.get("X-Filename") or "upload")[:200]
    m = EXT_RE.search(disp); ext = m.group(0).lower() if m else ".bin"
    os.makedirs(UPLOADS, exist_ok=True)
    tmp = os.path.join(UPLOADS, f".part-{secrets.token_hex(8)}"); sha = hashlib.sha256(); left = n
    try:
        with open(tmp, "wb") as f:
            while left:
                b = h.rfile.read(min(1 << 20, left))
                if not b: raise ApiError(400, "upload truncated")
                f.write(b); sha.update(b); left -= len(b)
        try: info = pipeline.probe(tmp)
        except ValueError as e: raise ApiError(400, f"not a usable video: {e}".replace(tmp, "upload"))
        name = sha.hexdigest()[:16] + ext; dst = os.path.join(UPLOADS, name)
        os.replace(tmp, dst); wjson(dst + ".json", dict(name=disp, probe=info, uploaded=time.time()))
        return dict(id="up-" + name, name=disp, probe=info)
    finally:
        if os.path.exists(tmp): os.remove(tmp)


def _int(d, k, default=None, lo=None, hi=None):
    v = d.get(k, default)
    if v is None or v == "": return default
    try: v = int(v)
    except (TypeError, ValueError): raise ApiError(400, f"{k} must be an integer")
    if (lo is not None and v < lo) or (hi is not None and v > hi): raise ApiError(400, f"{k} must be {lo}..{hi}")
    return v


def prep_args(d):
    a = dict(start=_int(d, "start", 0, 0, 10 ** 7), frames=_int(d, "frames", 24, 2, pipeline.LIMITS["max_frames"]),
             width=_int(d, "width", 32, pipeline.LIMITS["min_width"], pipeline.LIMITS["max_width"]), threshold=_int(d, "threshold", 128, 0, 255),
             invert=bool(d.get("invert", False)), fps=None)
    if d.get("fps") not in (None, ""):
        try: a["fps"] = float(d["fps"])
        except (TypeError, ValueError): raise ApiError(400, "fps must be a number")
        if not 0.1 <= a["fps"] <= 240: raise ApiError(400, "fps must be 0.1..240")
    return a


def do_prep(path, a):
    try: return pipeline.prep(path, a["start"], a["frames"], a["width"], a["fps"], a["threshold"], a["invert"])
    except ValueError as e: raise ApiError(400, str(e))


def b64npy(path):
    x = np.ascontiguousarray(np.load(path), np.uint8)
    return dict(b64=base64.b64encode(x.tobytes()).decode(), shape=list(x.shape))


def preview(d):
    a = prep_args(d); data, man = do_prep(video_path(d.get("video")), a)
    return dict(gray=b64npy(os.path.join(data, "gray.npy")), bin=b64npy(os.path.join(data, "bin.npy")), manifest=man)


# ---------- runs ----------
def run_dir(name):
    m = IMP_RE.match(name)
    if m: p = under(IMPORTED, m.group(1))
    elif RUN_RE.match(name) and ".." not in name: p = under(RUNS, name)
    else: raise ApiError(400, "bad run name")
    if not os.path.isdir(p): raise ApiError(404, "no such run")
    return p


def alive(name, job):
    if name in JOBS: return JOBS[name].poll() is None
    pid = (job or {}).get("pid")
    if not pid: return False
    try: os.kill(pid, 0)
    except ProcessLookupError: return False
    except PermissionError: return True
    try:  # a zombie is not alive
        with open(f"/proc/{pid}/stat") as f: return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError: return True


def gpuq_state(jid):
    """queued / allocating / running / finished, or None if gpuq cannot tell (the queue forgets jobs after -C)."""
    try: r = subprocess.run(["gpuq", "-s", str(jid)], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired): return None
    return r.stdout.strip() or None if r.returncode == 0 else None


def run_state(name, d):
    st = rjson(os.path.join(d, "status.json"), {}) or {}; job = rjson(os.path.join(d, "ui_job.json")); ui = rjson(os.path.join(d, "ui_status.json"))
    s = st.get("status") or ("finished" if os.path.exists(os.path.join(d, "model.ncab")) else "unknown")
    if ui and ui.get("status") == "cancelled" and s == "running": s = "cancelled"
    elif job and job.get("device") == "gpu":
        q = gpuq_state(job.get("gpuq_id"))
        if not st: s = "died" if q == "finished" else "queued" if q in ("queued", "allocating", None) else "starting"
        elif s == "running" and q == "finished": s = "died"
    elif job:
        live = alive(name, job)
        if not st: s = "starting" if live else "died"
        elif s == "running" and not live: s = "died"
    elif s == "running" and st.get("pid"):
        if not alive(name, dict(pid=st["pid"])): s = "died"
    return s, st, job, ui


def summary(name, d):
    man = rjson(os.path.join(d, "manifest.json"), {}) or {}; s, st, job, ui = run_state(name, d)
    ex = man.get("export") or {}; sizes = ex.get("sizes") or st.get("sizes") or {}
    prog = rjson(os.path.join(d, "train", "progress.json"), {}) or {}
    el = sum(v.get("seconds", 0) for v in (st.get("stages") or {}).values())
    if s in ("running", "starting") and job: el = time.time() - job["started"]
    pp = (man.get("data") or {}).get("preprocessing") or {}
    return dict(name=name, label=man.get("label") or (man.get("imported") or {}).get("label") or name, status=s, stage=st.get("stage"),
                kind="imported" if name.startswith("imp-") else "example" if name.startswith("example_") else "run",
                preset=man.get("preset"), n_params=ex.get("n_params") or sizes.get("n_params"), export_bytes=sizes.get("file"),
                has_export=any(os.path.exists(os.path.join(d, f)) for f in ("model.ncab", "model.nca.json", "model.json")),
                has_original=bool(man.get("data_dir")), elapsed_s=round(el, 1), own=bool(job), step=prog.get("step"), steps=prog.get("steps"),
                frames=pp.get("frames") or man.get("T"), width=pp.get("width") or man.get("W"), height=pp.get("height") or man.get("H"))


def list_runs():
    out = []
    for root, pre in ((RUNS, ""), (IMPORTED, "imp-")):
        if not os.path.isdir(root): continue
        for f in sorted(os.listdir(root)):
            name = pre + f
            if (IMP_RE.match(name) if pre else RUN_RE.match(name)) and os.path.isdir(os.path.join(root, f)):
                out.append(summary(name, os.path.join(root, f)))
    order = {"example": 0, "run": 1, "imported": 2}
    return sorted(out, key=lambda r: (order[r["kind"]], r["name"])) + (c12_list() if os.path.isdir(C12) else [])


def hist_of(d):
    h = rjson(os.path.join(d, "train", "hist.json"))
    if h is not None: return h
    try:  # live: hist.json is only written at the end; the resumable state carries it during training
        with np.load(os.path.join(d, "train", "state.npz")) as z: return json.loads(str(z["meta"]))["hist"]
    except (OSError, KeyError, ValueError): return []


def run_detail(name):
    d = run_dir(name); s, st, job, ui = run_state(name, d)
    log = []
    try:
        with open(os.path.join(d, "log.txt"), errors="replace") as f: log = f.read()[-20000:].splitlines()[-40:]
    except OSError: pass
    return dict(summary=summary(name, d), manifest=rjson(os.path.join(d, "manifest.json"), {}), status=st, job=job, ui_status=ui,
                progress=rjson(os.path.join(d, "train", "progress.json")), hist=hist_of(d), log_tail=log, presets=pipeline.PRESETS)


def env():
    e = dict(os.environ, JAX_PLATFORMS="cpu", OMP_NUM_THREADS="4", XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4")
    e.pop("CUDA_VISIBLE_DEVICES", None)
    return e


def launch(name, d, args, device, resume=False):
    cmd = [sys.executable, "-m", "ca.pipeline", "run"] + args + (["--resume"] if resume else [])
    job = dict(device=device, args=args, started=time.time(), resumed=resume)
    if device == "gpu":
        # the queue server's env is not ours: unset CPU pinning explicitly, no JAX preallocation (GPU is shared), and send
        # output to the run's log.txt (gpuq's own log is outside the run dir and invisible to the UI)
        log = os.path.join(d, "log.txt")
        wrapped = ["env", "-u", "JAX_PLATFORMS", "XLA_PYTHON_CLIENT_PREALLOCATE=false", "sh", "-c", 'exec "$@" >>"$0" 2>&1', log] + cmd
        r = subprocess.run(["gpuq", "-L", "ui-" + name, "-G", "1"] + wrapped, cwd=ROOT, capture_output=True, text=True, timeout=30)
        if r.returncode != 0 or not r.stdout.strip().split()[-1:][0].isdigit(): raise ApiError(500, f"gpuq failed: {r.stderr.strip() or r.stdout.strip()}")
        job["gpuq_id"] = r.stdout.strip().split()[-1]
    else:
        log = open(os.path.join(d, "log.txt"), "ab")
        p = subprocess.Popen(cmd, cwd=ROOT, start_new_session=True, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env())
        log.close(); JOBS[name] = p; job.update(pid=p.pid, pgid=os.getpgid(p.pid))
    wjson(os.path.join(d, "ui_job.json"), job)
    if os.path.exists(os.path.join(d, "ui_status.json")): os.remove(os.path.join(d, "ui_status.json"))
    return job


def start_train(d):
    path = video_path(d.get("video")); a = prep_args(d)
    preset = d.get("preset", "demo")
    if preset not in pipeline.PRESETS: raise ApiError(400, f"preset must be one of {sorted(pipeline.PRESETS)}")
    steps = _int(d, "steps", None, 1, 200000); device = d.get("device", "cpu")
    if device not in ("cpu", "gpu"): raise ApiError(400, "device must be cpu or gpu")
    do_prep(path, a)  # validate interval now (cached; the run reuses it)
    name = f"ui_{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(2)}"; rd = os.path.join(RUNS, name); os.makedirs(rd)
    args = [path, rd, "--start", str(a["start"]), "--frames", str(a["frames"]), "--width", str(a["width"]), "--threshold", str(a["threshold"]), "--preset", preset]
    if a["fps"]: args += ["--fps", str(a["fps"])]
    if a["invert"]: args += ["--invert"]
    if steps: args += ["--steps", str(steps)]
    return dict(name=name, job=launch(name, rd, args, device))


def cancel(name):
    d = run_dir(name); job = rjson(os.path.join(d, "ui_job.json"))
    if not job: raise ApiError(403, "only runs started by this UI can be cancelled")
    if job.get("device") == "gpu":
        q = gpuq_state(job["gpuq_id"])
        if q == "finished": raise ApiError(409, "job is not running")
        # -r removes a waiting job, -k sends SIGTERM to a running one; both only ever for this run's own job id
        subprocess.run(["gpuq", "-k" if q == "running" else "-r", str(job["gpuq_id"])], cwd=ROOT, capture_output=True, timeout=30)
    elif alive(name, job):
        try: os.killpg(job["pgid"], signal.SIGTERM)
        except ProcessLookupError: pass
        if name in JOBS:
            try: JOBS[name].wait(10)
            except subprocess.TimeoutExpired: os.killpg(job["pgid"], signal.SIGKILL); JOBS[name].wait(5)
    else: raise ApiError(409, "job is not running")
    wjson(os.path.join(d, "ui_status.json"), dict(status="cancelled", at=time.time()))
    return dict(name=name, cancelled=True)


def resume(name):
    d = run_dir(name); job = rjson(os.path.join(d, "ui_job.json"))
    if not job: raise ApiError(403, "only runs started by this UI can be resumed")
    s = run_state(name, d)[0]
    if s in ("running", "starting", "queued"): raise ApiError(409, f"run is {s}")
    if s == "finished": raise ApiError(409, "run already finished")
    return dict(name=name, job=launch(name, d, job["args"], job.get("device", "cpu"), resume=True))


def export_file(name):
    d = run_dir(name)
    for f, ct in (("model.ncab", "application/octet-stream"), ("model.nca.json", "application/json"), ("model.json", "application/json")):
        if os.path.exists(os.path.join(d, f)): return os.path.join(d, f), ct, f"{name}{'.ncab' if f.endswith('.ncab') else '.nca.json'}"
    raise ApiError(404, "no export yet")


def original(name):
    man = rjson(os.path.join(run_dir(name), "manifest.json"), {}) or {}
    if not man.get("data_dir"): raise ApiError(404, "no original frames for this model (CA-only playback)")
    d = under(CACHE, os.path.basename(man["data_dir"]))
    out = b64npy(os.path.join(d, "bin.npy")); pp = man["data"]["preprocessing"]
    return dict(bin=out, fps=pp["fps"], source_frame_index=pp["source_frame_index"], source_time_s=pp["source_time_s"])


def import_export(h):
    n = int(h.headers.get("Content-Length") or -1)
    if n <= 0: raise ApiError(411, "Content-Length required")
    if n > MAX_EXPORT: raise ApiError(413, f"export larger than {MAX_EXPORT >> 20} MB")
    data = h.rfile.read(n)
    if len(data) != n: raise ApiError(400, "upload truncated")
    ext = ".ncab" if data[:4] == b"NCAB" else ".json"
    sha = hashlib.sha256(data).hexdigest()[:16]; d = os.path.join(IMPORTED, sha); os.makedirs(IMPORTED, exist_ok=True)
    tmpd = os.path.join(IMPORTED, f".part-{secrets.token_hex(8)}"); os.makedirs(tmpd)
    try:
        f = os.path.join(tmpd, "model" + (".ncab" if ext == ".ncab" else ".json"))
        with open(f, "wb") as fh: fh.write(data)
        try: hd, P, f0 = pipeline.load_export(f)
        except Exception as e: raise ApiError(400, f"not a valid NCA export (.ncab or nca-v1 JSON): {type(e).__name__}: {e}")
        n_params = int(sum(np.asarray(v).size for v in P.values()))
        label = (h.headers.get("X-Filename") or "imported export")[:120]
        man = dict(label=label, T=hd["T"], H=int(f0.shape[0]), W=int(f0.shape[1]), C=hd["C"], Fh=hd["Fh"], S=hd["S"], imported=dict(meta=hd.get("meta", {}), label=label),
                   export=dict(n_params=n_params, sizes=dict(file=len(data), n_params=n_params), weight_dtype=hd.get("weight_dtype", "f32")))
        wjson(os.path.join(tmpd, "manifest.json"), man)
        wjson(os.path.join(tmpd, "status.json"), dict(status="finished", stage=None, error=None, imported=True))
        if os.path.isdir(d): return dict(name="imp-" + sha, existing=True, **man)
        os.replace(tmpd, d)
        return dict(name="imp-" + sha, existing=False, **man)
    finally:
        if os.path.isdir(tmpd):
            for x in os.listdir(tmpd): os.remove(os.path.join(tmpd, x))
            os.rmdir(tmpd)


# ---------- runs/c12 exports (read-only listing; CA frames are still computed in the browser) ----------
def c12_path(name):
    if not C12_RE.match(name) or ".." in name: raise ApiError(400, "bad c12 name")
    p = under(C12, *name.split("~")[1:])
    if not (p.endswith(".ncab") or p.endswith(".rule.json")) or not os.path.isfile(p): raise ApiError(404, "no such c12 export")
    return p


def c12_header(p):
    """Header of an ncab (v1/v2) or the dlrule-v1 JSON without the packed frames; raises on anything else."""
    import zlib
    with open(p, "rb") as f: data = f.read(MAX_EXPORT + 1)
    if len(data) > MAX_EXPORT: raise ValueError("larger than the export limit")
    if p.endswith(".rule.json"):
        o = json.loads(data)
        if not (o.get("format") in ("ca-universal/dlrule-v1", "ca-universal/dlrule-v2") or (o.get("format") == "ca-universal/rule-v1" and isinstance(o.get("programs"), list))): raise ValueError("not dlrule-v1/v2")
        return dict({k: v for k, v in o.items() if k not in ("first", "h0")}, format="dlrule"), len(data)
    if data[:4] != b"NCAB" or data[4] not in (1, 2): raise ValueError("not ncab-v1/v2")
    body = zlib.decompress(data[6:]) if data[5] & 1 else data[6:]; hl = int.from_bytes(body[:4], "little")
    return dict(json.loads(body[4:4 + hl]), format=f"ncab-v{data[4]}"), len(data)


def c12_start(meta):
    """Source start frame from export meta: src_start (exp45 NCA exports) or spec 'start:T' (rule exports)."""
    if meta.get("src_start") is not None: return int(meta["src_start"])
    m = re.match(r"^(\d+):\d+$", str(meta.get("spec", "")))
    return int(m.group(1)) if m else None


def c12_summary(name, p):
    try: hd, n = c12_header(p)
    except (OSError, ValueError) as e: return dict(name=name, label=os.path.relpath(p, C12), status="failed", kind="c12", error=str(e), has_export=False)
    if hd["format"] == "dlrule":
        from ca.rules import resolve
        npar = None; extra = dict(rules=sum(len(q["rules"]) for q in resolve(hd["programs"])), programs=len(hd["programs"]))
    else:
        shapes = [s for _, s in hd.get("blobs") or hd.get("params") or []]; npar = int(sum(np.prod(s) for s in shapes)); extra = dict(phases=len(hd.get("schedule") or [1]), mode=hd.get("mode", "v1"))
    meta = hd.get("meta") or {}
    return dict(name=name, label=os.path.relpath(p, C12), status="finished", kind="c12", format=hd["format"], n_params=npar, export_bytes=n, has_export=True,
                has_original=c12_start(meta) is not None, frames=hd["T"], width=hd["W"], height=hd["H"], own=False, elapsed_s=None, **extra)


def c12_list():
    out = []
    for dp, dns, fns in os.walk(C12):
        dns.sort()
        for f in sorted(fns):
            if f.endswith(".ncab") or f.endswith(".rule.json"):
                name = "c12~" + os.path.relpath(os.path.join(dp, f), C12).replace(os.sep, "~")
                if C12_RE.match(name): out.append(c12_summary(name, os.path.join(dp, f)))
    return out


def c12_detail(name):
    p = c12_path(name); sm = c12_summary(name, p); hd = c12_header(p)[0] if sm["status"] == "finished" else {}
    return dict(summary=sm, manifest=dict(label=sm["label"], c12=dict(path=os.path.relpath(p, ROOT), header={k: v for k, v in hd.items() if k != "programs"}),
                export=dict(sizes=dict(file=sm.get("export_bytes")), n_params=sm.get("n_params"))),
                status=dict(status=sm["status"], error=sm.get("error")), job=None, ui_status=None, progress=None, hist=[], log_tail=[], presets=pipeline.PRESETS)


def c12_original(name):
    return export_original(c12_path(name))


def export_original(p):
    """Original binary frames for an ncab/dlrule export from its meta (src_start or spec + T frames of the source at meta.width,
    threshold meta.threshold, via ca.pipeline.prep). Comparison only; first_frame_mismatch_px reports whether it matches the export."""
    hd = c12_header(p)[0]; meta = hd.get("meta") or {}
    start = c12_start(meta)
    if start is None: raise ApiError(404, "export meta has no src_start/spec: CA-only playback")
    src = under(ROOT, meta["source"]) if meta.get("source") else BUILTIN["badapple"]
    if not os.path.isfile(src): raise ApiError(404, "source video of this export is not available")
    a = dict(start=start, frames=int(hd["T"]), width=int(meta.get("width", hd["W"])), threshold=int(meta.get("threshold", 128)), invert=False, fps=None)
    d, man = do_prep(src, a); F = np.load(os.path.join(d, "bin.npy"))
    if F.shape[1:] != (hd["H"], hd["W"]): raise ApiError(409, f"prepared original is {F.shape[1:]} but the export is {(hd['H'], hd['W'])}")
    if hd["format"] == "dlrule":
        from ca import rules
        f0 = rules.load(p)["first"]
    elif hd["format"] == "ncab-v2":
        from ca import phase_nca
        f0 = phase_nca.load(p)[4]
    else: f0 = pipeline.load_export(p)[2]
    out = b64npy(os.path.join(d, "bin.npy")); pp = man["preprocessing"]
    return dict(bin=out, fps=pp["fps"], source_frame_index=pp["source_frame_index"], source_time_s=pp["source_time_s"], first_frame_mismatch_px=int((F[0] != f0).sum()))


# ---------- featured model (runs/featured/manifest.json, chosen by the experiment owner; never hard-coded here) ----------
FEATURED = os.path.join(ROOT, "runs", "featured", "manifest.json")


def featured_path():
    man = rjson(FEATURED)
    if not man or not (man.get("featured") or {}).get("export"): raise ApiError(404, "runs/featured/manifest.json has no featured.export")
    p = under(os.path.join(ROOT, "runs"), man["featured"]["export"].removeprefix("runs/"))
    if not p.endswith(".ncab") or not os.path.isfile(p): raise ApiError(404, f"featured export {man['featured']['export']} not found")
    return man, p


def featured():
    man, p = featured_path(); f = man["featured"]
    with open(p, "rb") as fh: sha = hashlib.sha256(fh.read()).hexdigest()
    hd = c12_header(p)[0]
    return dict(manifest=man, path=os.path.relpath(p, ROOT), file_sha256=sha, sha_ok=sha == f.get("sha256"),
                header={k: v for k, v in hd.items() if k not in ("programs",)}, fps=f.get("fps") or (hd.get("meta") or {}).get("fps") or 30)


# ---------- HTTP ----------
class H(BaseHTTPRequestHandler):
    server_version = "ca-ui/1"

    def log_message(self, fmt, *a): sys.stderr.write("%s %s\n" % (self.command, fmt % a))

    def send(self, code, body, ct="application/json", extra=None):
        if not isinstance(body, (bytes, bytearray)): body = json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ct); self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store"); self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.end_headers(); self.wfile.write(body)

    def file(self, p, ct=None, dl=None):
        if not os.path.isfile(p): raise ApiError(404, "not found")
        with open(p, "rb") as f: b = f.read()
        self.send(200, b, ct or MIME.get(os.path.splitext(p)[1], "application/octet-stream"), {"Content-Disposition": f'attachment; filename="{dl}"'} if dl else None)

    def body_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 1 << 20: raise ApiError(413, "request too large")
        try: return json.loads(self.rfile.read(n) or b"{}")
        except ValueError: raise ApiError(400, "body must be JSON")

    def guard(self):
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        if host not in ("127.0.0.1", "localhost", "[::1]", self.server.server_address[0]): raise ApiError(403, "bad Host header")
        # custom header forces a CORS preflight, so other web pages cannot POST here
        if self.command == "POST" and self.headers.get("X-CA-UI") != "1": raise ApiError(403, "missing X-CA-UI header")

    def route(self):
        self.guard(); path = urlparse(self.path).path; parts = [p for p in path.split("/") if p]; G = self.command == "GET"
        if G and path in ("/", "/index.html"): return self.file(os.path.join(STATIC, "index.html"))
        if G and parts[:1] == ["static"] and len(parts) >= 2: return self.file(under(STATIC, *parts[1:]))
        if G and len(parts) >= 3 and parts[0] == "assets" and parts[1] in ("logo", "figures") and parts[-1].endswith(".png"):
            return self.file(under(os.path.join(ROOT, "assets", parts[1]), *parts[2:]))  # read-only: logo + figures, png only
        if G and parts[:1] == ["player"] and len(parts) == 2 and parts[1].endswith(".mjs"): return self.file(under(PLAYER, parts[1]))
        if parts[:1] != ["api"]: raise ApiError(404, "not found")
        p = parts[1:]
        if G and p == ["config"]: return self.send(200, dict(presets=pipeline.PRESETS, limits=pipeline.LIMITS, max_upload=MAX_UPLOAD, max_export=MAX_EXPORT))
        if G and p == ["videos"]: return self.send(200, list_videos())
        if G and len(p) == 3 and p[0] == "video" and p[2] == "probe":
            try: return self.send(200, pipeline.probe(video_path(p[1])))
            except ValueError as e: raise ApiError(400, str(e))
        if not G and p == ["upload"]: return self.send(200, upload(self))
        if not G and p == ["preview"]: return self.send(200, preview(self.body_json()))
        if not G and p == ["train"]: return self.send(200, start_train(self.body_json()))
        if not G and p == ["import-export"]: return self.send(200, import_export(self))
        if G and p == ["runs"]: return self.send(200, list_runs())
        if G and p == ["featured"]: return self.send(200, featured())
        if G and p == ["featured", "export"]: f = featured_path()[1]; return self.file(f, "application/octet-stream", os.path.basename(f))
        if G and p == ["featured", "original"]: return self.send(200, export_original(featured_path()[1]))
        if len(p) >= 2 and p[0] == "runs":
            name = p[1]; rest = p[2:]
            if name.startswith("c12~") and G:
                if not rest: return self.send(200, c12_detail(name))
                if rest == ["export"]: f = c12_path(name); return self.file(f, "application/octet-stream", os.path.basename(f))
                if rest == ["original"]: return self.send(200, c12_original(name))
                raise ApiError(404, "not found")
            if G and not rest: return self.send(200, run_detail(name))
            if G and rest in (["export"], ["export.ncab"]): return self.file(*export_file(name))
            if G and rest == ["original"]: return self.send(200, original(name))
            if not G and rest == ["cancel"]: return self.send(200, cancel(name))
            if not G and rest == ["resume"]: return self.send(200, resume(name))
        raise ApiError(404, "not found")

    def handle_any(self):
        try: self.route()
        except ApiError as e: self.send(e.code, dict(error=str(e)))
        except Exception as e:
            self.log_error("internal error: %r", e); self.send(500, dict(error=f"internal error: {type(e).__name__}: {e}"))

    do_GET = do_POST = handle_any


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(); os.makedirs(UPLOADS, exist_ok=True); os.makedirs(IMPORTED, exist_ok=True)
    srv = ThreadingHTTPServer((a.host, a.port), H)
    print(f"Bad CA!! on http://{a.host}:{a.port}/  (Ctrl-C to stop; CPU training jobs keep running in their own session)", flush=True)
    try: srv.serve_forever()
    except KeyboardInterrupt: pass
    finally: srv.server_close()


if __name__ == "__main__":
    main()
