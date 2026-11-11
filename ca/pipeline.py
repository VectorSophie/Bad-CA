"""Video -> binary grid -> Neural CA training -> export -> independent replay -> comparison video.

CLI (python -m ca.pipeline <cmd> ...):
  probe  VIDEO                                   decoder-validated metadata (JSON)
  prep   VIDEO --start S --frames N --width W [--fps F] [--threshold T] [--invert]
                                                 -> cache/pipe/<key>/ (gray.npy, bin.npy, manifest.json); key = sha256 of
                                                    (source sha256, preprocessing config); only the requested interval is decoded
  train  RUN --data DATA_DIR [--preset demo|quality] [--C --Fh --steps --lr --lamB --seeds] [--resume]
  export RUN [--f16]                             best checkpoint -> RUN/model.ncab (+ model.nca.json), manifest
  replay EXPORT                                  Python replay vs independent Node replay (binary mismatches, |dv|)
  video  EXPORT --data DATA_DIR [--out STEM]     original / CA / diff MP4 (1x, 0.25x) + GIF, decoded back
  run    VIDEO RUN [prep + train + export + replay + video options]   the whole chain; RUN/status.json tracks stages
  import EXPORT RUN --video V --start S --width W   ready-to-play run dir from an existing export (no training; ncab-v2 copied as is)
  info   EXPORT                                  header summary of nca-v1 / ncab-v1 / ncab-v2 (phases, schedule, params, bytes)
  train/run ... --phases uniform:K|auto:K [--noise SIGMA] [--max-hours H]
                                                 phased NCA (time-varying CA, external schedule, state carried) -> ncab-v2
Preprocessing (recorded in every manifest): frames decoded with PyAV; colour -> luma by PIL convert("L")
(ITU-R 601-2: L = 0.299 R + 0.587 G + 0.114 B); box resize to `width`, height = round(width * h / w) (aspect kept);
binary = gray >= threshold (or < with --invert). Output fps: frame k is the source frame whose timestamp is nearest
to start + k / fps (no blending); default fps = source fps (every frame). Start is in source frames (--start) or seconds
(--start-sec). Training/export/replay never touch the source video; replay uses the export alone.
Presets are explicit: user flags override them and the effective values are written to the manifest.
Limits (local demo, not a public service): <= 2 GB file, <= 4096 px source side, interval <= 8000 frames (the whole 6,572-frame Bad Apple fits), width 8..256."""
import argparse, hashlib, json, os, subprocess, sys, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(ROOT, "cache", "pipe")
PREPROC_VERSION = 1
LIMITS = dict(max_bytes=2 << 30, max_side=4096, max_frames=8000, min_width=8, max_width=256)
PRESETS = {
    "demo": dict(C=8, Fh=16, steps=1500, lr=2e-3, lr_end=2e-5, lamB=2.0, seeds=1,
                 note="fast CPU demo: 1,304 params; meant for <= 32 px wide and <= 24 frames; quality far below the campaign models"),
    "quality": dict(C=16, Fh=64, steps=20000, lr=2e-3, lr_end=2e-5, lamB=2.0, seeds=1,
                    note="C10-size model (10,320 params) with the C10 contour loss; needs a GPU (minutes to tens of minutes)"),
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def _atomic_json(path, obj):
    tmp = path + ".tmp"; json.dump(obj, open(tmp, "w"), indent=1, default=float); os.replace(tmp, path)


def probe(path):
    """Open with the decoder (never trust the extension); raises ValueError with an actionable message."""
    import av
    if not os.path.isfile(path): raise ValueError(f"no such file: {path}")
    if os.path.getsize(path) > LIMITS["max_bytes"]: raise ValueError("file larger than 2 GB")
    try:
        with av.open(path) as c:
            if not c.streams.video: raise ValueError("no video stream in file")
            s = c.streams.video[0]; w, h = s.codec_context.width, s.codec_context.height
            fps = float(s.average_rate) if s.average_rate else None
            dur = float(s.duration * s.time_base) if s.duration else (c.duration / 1e6 if c.duration else None)
            first = next(c.decode(s), None); n_frames = s.frames
            if first is None: raise ValueError("decoder produced no frames")
    except Exception as e:  # PyAV/FFmpeg errors (some subclass ValueError) -> one readable message; keep our own messages
        if isinstance(e, ValueError) and not isinstance(e, av.error.FFmpegError): raise
        raise ValueError(f"cannot decode as video: {type(e).__name__}: {e}")
    if not w or not h or max(w, h) > LIMITS["max_side"]: raise ValueError(f"unsupported frame size {w}x{h} (max side {LIMITS['max_side']})")
    if not fps: raise ValueError("stream has no frame rate")
    frames = n_frames or (round(dur * fps) if dur else None)
    return dict(width=w, height=h, fps=fps, frames=frames, duration_s=dur, codec=s.codec_context.name, pix_fmt=first.format.name, bytes=os.path.getsize(path))


def prep(video, start=0, frames=24, width=32, fps=None, threshold=128, invert=False, start_sec=None, src_sha=None):
    """Decode only the requested interval, cache gray + binary grids. Returns (data_dir, manifest)."""
    import av
    from PIL import Image
    info = probe(video); fps_out = float(fps or info["fps"])
    if not (LIMITS["min_width"] <= width <= LIMITS["max_width"]): raise ValueError(f"width must be {LIMITS['min_width']}..{LIMITS['max_width']}")
    if not (2 <= frames <= LIMITS["max_frames"]): raise ValueError(f"frames must be 2..{LIMITS['max_frames']}")
    if not (0 <= threshold <= 255): raise ValueError("threshold must be 0..255")
    t_start = float(start_sec) if start_sec is not None else start / info["fps"]
    if info["duration_s"] and t_start + (frames - 1) / fps_out > info["duration_s"] + 1e-6:
        raise ValueError(f"interval ends at {t_start + (frames - 1) / fps_out:.2f}s but the video lasts {info['duration_s']:.2f}s")
    src_sha = src_sha or sha256_file(video)
    cfg = dict(v=PREPROC_VERSION, src_sha256=src_sha, t_start=round(t_start, 6), frames=frames, width=width, fps=fps_out, threshold=threshold, invert=bool(invert))
    key = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]; d = os.path.join(CACHE, key)
    if os.path.exists(os.path.join(d, "manifest.json")): return d, json.load(open(os.path.join(d, "manifest.json")))
    want = [t_start + k / fps_out for k in range(frames)]; got = [None] * frames; src_idx = [None] * frames; best = [np.inf] * frames
    height = round(width * info["height"] / info["width"]); half = 0.5 / info["fps"]
    with av.open(video) as c:
        s = c.streams.video[0]
        for i, fr in enumerate(c.decode(s)):   # shortcut: sequential decode from the start; add keyframe seek for long remote files
            ts = float(fr.pts * s.time_base) if fr.pts is not None else i / info["fps"]
            if ts > want[-1] + 2 * half: break
            k = int(round((ts - t_start) * fps_out))   # the one output slot this frame can be nearest to
            if 0 <= k < frames and abs(ts - want[k]) < best[k]:
                best[k] = abs(ts - want[k]); src_idx[k] = i
                got[k] = np.asarray(fr.to_image().convert("L").resize((width, height), Image.BOX))
    if any(g is None for g in got) or max(best) > half + 1e-6: raise ValueError("could not find source frames for the whole interval (check start/frames/fps)")
    gray = np.stack(got); binary = ((gray < threshold) if invert else (gray >= threshold)).astype(np.uint8)
    warn = []   # a uniform first frame is allowed (Bad Apple starts black); only the zero boundary or stored initial hidden can break symmetry
    if binary[0].min() == binary[0].max(): warn.append("first frame is uniform after thresholding: structure can only come from the zero boundary or a stored initial hidden state (phased runs may learn one)")
    os.makedirs(d, exist_ok=True); np.save(os.path.join(d, "gray.npy"), gray); np.save(os.path.join(d, "bin.npy"), binary)
    man = dict(source=dict(path=os.path.abspath(video), sha256=src_sha, **info), preprocessing=dict(cfg, height=height, luma="PIL convert('L'): ITU-R 601-2 luma",
               resize="PIL BOX", binarize=("gray < threshold" if invert else "gray >= threshold"), frame_selection="nearest source timestamp to t_start + k/fps",
               source_frame_index=src_idx, source_time_s=[round(t, 6) for t in want]), key=key, shape=list(binary.shape), created=time.time(), warnings=warn)
    _atomic_json(os.path.join(d, "manifest.json"), man)
    return d, man


def _phases(spec, F):
    from ca import segment
    kind, K = spec.split(":"); K = int(K)
    if kind == "uniform": return segment.uniform_k(len(F), K)
    if kind == "auto": return segment.best_k(F, K, lmin=min(8, (len(F) - 1) // K))["phases"]
    raise ValueError("--phases must be uniform:K or auto:K")


def train_run(run, data, preset="demo", resume=False, phases=None, noise=None, max_hours=None, **over):
    from ca.nca_train import train
    cfg = dict(PRESETS[preset]); cfg.update({k: v for k, v in over.items() if v is not None})
    if phases:   # phased NCA (time-varying CA, external schedule): chunked-joint training with numpy-carried state, ncab-v2 export
        from ca import phase_nca
        from ca.phase_train import chunked
        F = np.load(os.path.join(data, "bin.npy")); dman = json.load(open(os.path.join(data, "manifest.json"))); os.makedirs(run, exist_ok=True)
        ph = _phases(phases, F); cfg.update(phases=phases, noise=noise or 0.0, max_hours=max_hours, n_phases=len(ph))
        _atomic_json(os.path.join(run, "manifest.json"), dict(data_dir=os.path.abspath(data), data=dman, preset=preset, settings=cfg, phased=True,
                     regime="phased: chunked-joint BPTT, state carried across phases and chunks (numpy carry), training-only state noise"))
        P, g, used = chunked(F, ph, os.path.join(run, "train"), seed=0, C=int(cfg["C"]), Fh=int(cfg["Fh"]), S=4, steps=int(cfg["steps"]), sched=("cos", cfg["lr"], cfg["lr_end"]),
                             init_factor=None, eval_every=max(10, int(cfg["steps"]) // 30), noise=noise or 0.0, cpu_carry=True, max_hours=max_hours,
                             chunk_min=min(200, len(F) - 1), log=lambda s: print(s, flush=True))
        T = used[-1]["end"]; pp = dman["preprocessing"]
        meta = dict(source_sha256=dman["source"]["sha256"], source_frames=[pp["source_frame_index"][0], pp["source_frame_index"][T - 1]], fps=pp["fps"], width=pp["width"],
                    threshold=pp["threshold"], invert=pp["invert"], data_key=dman["key"], partial=T < len(F), phases=phases, noise=noise or 0.0)
        out = os.path.join(run, "model.ncab"); sizes = phase_nca.export(out, P, "indep", used, F[0], T, 4, meta=meta)
        man = json.load(open(os.path.join(run, "manifest.json"))); man["export"] = dict(file=out, sizes=sizes, weight_dtype="f32", n_params=sizes["n_params"], format="ncab-v2")
        _atomic_json(os.path.join(run, "manifest.json"), man)
        return dict(best=json.load(open(os.path.join(run, "train", "chunks.json")))["chunks"][-1]["best"], export=out, phases=len(used), frames=T)
    F = np.load(os.path.join(data, "bin.npy")); dman = json.load(open(os.path.join(data, "manifest.json")))
    os.makedirs(run, exist_ok=True)
    man = dict(data_dir=os.path.abspath(data), data=dman, preset=preset, settings=cfg, regime="autonomous from frame 0, hidden 0, BPTT over the whole interval")
    _atomic_json(os.path.join(run, "manifest.json"), man)
    models = [dict(name=f"s{s}", seed=s, lamB=cfg["lamB"], lamT=0.0) for s in range(int(cfg["seeds"]))]
    return train(F, models, os.path.join(run, "train"), C=int(cfg["C"]), Fh=int(cfg["Fh"]), S=4, steps=int(cfg["steps"]), sched=("cos", cfg["lr"], cfg["lr_end"]),
                 eval_every=max(10, int(cfg["steps"]) // 30), resume=resume, meta=dict(run=run), log=lambda s: print(s, flush=True))


def export_run(run, f16=False):
    from ca import nca, ncab
    man = json.load(open(os.path.join(run, "manifest.json")))
    if man.get("phased"): return man["export"]["file"], man["export"]["sizes"]   # ncab-v2 written by train_run
    tr = json.load(open(os.path.join(run, "train", "train.json")))
    name = min(tr["best"], key=lambda k: tr["best"][k]["err"]); P = dict(np.load(os.path.join(run, "train", f"best_{name}.npz")))
    F = np.load(os.path.join(man["data_dir"], "bin.npy")); pp = man["data"]["preprocessing"]
    meta = dict(source_sha256=man["data"]["source"]["sha256"], source_frames=[pp["source_frame_index"][0], pp["source_frame_index"][-1]], fps=pp["fps"], width=pp["width"],
                threshold=pp["threshold"], invert=pp["invert"], data_key=man["data"]["key"], model=name, train_err=tr["best"][name]["err"], steps=tr["steps"])
    out = os.path.join(run, "model.ncab"); sizes = ncab.write(out, P, F[0], len(F), 4, meta, "f16" if f16 else "f32")
    nca.export(os.path.join(run, "model.nca.json"), P, F[0], len(F), 4, meta)
    man["export"] = dict(file=out, json=os.path.join(run, "model.nca.json"), sizes=sizes, weight_dtype="f16" if f16 else "f32", n_params=sizes["n_params"])
    _atomic_json(os.path.join(run, "manifest.json"), man)
    return out, sizes


def is_v2(path):
    with open(path, "rb") as f: return f.read(5) == b"NCAB\x02"


def info(path):
    """Header summary of any export (nca-v1 JSON, ncab-v1, ncab-v2 phased)."""
    if is_v2(path):
        from ca import phase_nca
        hd, E, sched, h0, f0 = phase_nca.load(path)
        return dict(format="ncab-v2", **{k: hd[k] for k in ("T", "H", "W", "C", "Fh", "S", "mode", "K", "init", "weight_dtype")}, phases=len(hd["schedule"]),
                    schedule=hd["schedule"], n_params=int(sum(v.size for v in E.values())), bytes=os.path.getsize(path), meta=hd.get("meta", {}))
    hd, P, f0 = load_export(path)
    return dict(format="ncab-v1" if path.endswith(".ncab") else "nca-v1", T=hd["T"], H=int(f0.shape[0]), W=int(f0.shape[1]), S=hd["S"], K=1,
                n_params=int(sum(v.size for v in P.values())), bytes=os.path.getsize(path), meta=hd.get("meta", {}))


def replay_arrays(path):
    """(binary frames, continuous visible, first frame, T) replayed from the export alone (numpy reference)."""
    if is_v2(path):
        from ca import phase_nca
        hd, E, sched, h0, f0 = phase_nca.load(path); B, V = phase_nca.rollout_np(E, f0, sched, hd["S"], h0); return B, V, f0, hd["T"]
    from ca import nca
    hd, P, f0 = load_export(path); B, V = nca.rollout_np(P, f0[None], hd["T"], hd["S"]); return B, V, f0, hd["T"]


def load_export(path):
    """(header, float32 params, first frame) from .ncab (v1) or nca-v1 .json. Phased ncab-v2: use info()/replay_arrays()."""
    if path.endswith(".ncab") and is_v2(path): raise ValueError("ncab-v2 (phased) file: single-rule loader does not apply; use replay/info")
    from ca import nca, ncab
    if path.endswith(".ncab"): return ncab.read(path)
    ex, P, f0 = nca.load(path); return dict(T=ex["n_frames"], S=ex["S"], C=ex["C"], Fh=ex["Fh"], H=ex["height"], W=ex["width"], meta=ex.get("meta", {})), P, f0


def replay(path, node=True):
    """Replay from the export alone (numpy) and, if node, the independent Node player; report agreement."""
    B, V, f0, T = replay_arrays(path)
    rep = dict(T=T, H=int(f0.shape[0]), W=int(f0.shape[1]), format="ncab-v2" if is_v2(path) else "v1")
    if node:
        tmp = path + ".node.f32"
        o = json.loads(subprocess.run(["node", os.path.join(ROOT, "player", "nca_replay.mjs"), path, tmp], capture_output=True, text=True, check=True).stdout)
        Vn = np.fromfile(tmp, np.float32).reshape(V.shape); os.remove(tmp); Bn = np.concatenate([f0[None], Vn[1:] > 0.5]).astype(np.uint8)
        d = np.abs(Vn[1:] - V[1:]); rep.update(binary_mismatch_px=int((Bn != B).sum()), max_abs_dv=float(d.max()), mean_abs_dv=float(d.mean()),
                                              max_abs_dv_per_frame=[float(x) for x in d.reshape(len(d), -1).max(1)], node_sha256=o["sha256"],
                                              numpy_sha256=hashlib.sha256(np.ascontiguousarray(B, np.uint8).tobytes()).hexdigest())
    return B, V, rep


def video(path, data, stem):
    from ca.video_out import diff_rgb, make_video
    B, V, rep = replay(path, node=False); F = np.load(os.path.join(data, "bin.npy"))[:len(B)]
    man = json.load(open(os.path.join(data, "manifest.json"))); fps = man["preprocessing"]["fps"]
    err = float((B[1:] != F[1:]).mean())
    ch = make_video([("original (binary)", F), ("CA replay (export only)", B), ("diff red FN blue FP", diff_rgb(B, F))],
                    [f"autonomous replay from the export: first frame + hidden 0, {len(B)} frames, no target injection", f"pixel error {err:.4f}"],
                    lambda t: f"frame {t}  t={t / fps:5.2f}s", stem, fps_list=((fps, "1x"), (fps / 4, "0.25x")), scale=max(1, 288 // F.shape[2]))
    return ch


def import_export(export, run, video, start, width, threshold=128, invert=False, label=None, f16=False):
    """Make a ready-to-play run dir from an existing export (no training): converts it to ncab, prepares the matching
    original frames (comparison only) and checks that they start with the export's first frame."""
    from ca import ncab
    if is_v2(export):   # phased: copied unchanged (no conversion), original prepared for comparison only
        import shutil
        inf = info(export); T = inf["T"]
        from ca import phase_nca
        f0 = phase_nca.load(export)[4]; d, dman = prep(video, start, T, width, None, threshold, invert); F = np.load(os.path.join(d, "bin.npy"))
        if F.shape[1:] != f0.shape or not (F[0] == f0).all(): raise ValueError("prepared original does not start with the export's first frame (check start/width/threshold)")
        os.makedirs(run, exist_ok=True); shutil.copyfile(export, os.path.join(run, "model.ncab"))
        sizes = dict(file=os.path.getsize(export), n_params=inf["n_params"], phases=inf["phases"], K=inf["K"], format="ncab-v2")
        _atomic_json(os.path.join(run, "manifest.json"), dict(data_dir=os.path.abspath(d), data=dman, imported=dict(inf["meta"], imported_from=os.path.relpath(export, ROOT), label=label),
                     label=label, export=dict(file=os.path.join(run, "model.ncab"), sizes=sizes, weight_dtype=inf["weight_dtype"], n_params=inf["n_params"], format="ncab-v2")))
        _atomic_json(os.path.join(run, "status.json"), dict(status="finished", stage=None, error=None, export=os.path.join(run, "model.ncab"), sizes=sizes, imported=True))
        return run, sizes
    hd, P, f0 = load_export(export); d, dman = prep(video, start, hd["T"], width, None, threshold, invert)
    F = np.load(os.path.join(d, "bin.npy"))
    if F.shape[1:] != f0.shape or not (F[0] == f0).all(): raise ValueError("prepared original does not start with the export's first frame (check start/width/threshold)")
    os.makedirs(run, exist_ok=True); meta = dict(hd.get("meta", {}), imported_from=os.path.relpath(export, ROOT), label=label)
    sizes = ncab.write(os.path.join(run, "model.ncab"), P, f0, hd["T"], hd["S"], dict(source_sha256=dman["source"]["sha256"], label=label, imported_from=meta["imported_from"]), "f16" if f16 else "f32")
    import shutil; shutil.copyfile(export, os.path.join(run, "model.nca.json")) if export.endswith(".json") else None
    _atomic_json(os.path.join(run, "manifest.json"), dict(data_dir=os.path.abspath(d), data=dman, imported=meta, label=label,
                 export=dict(file=os.path.join(run, "model.ncab"), sizes=sizes, weight_dtype="f16" if f16 else "f32", n_params=sizes["n_params"])))
    _atomic_json(os.path.join(run, "status.json"), dict(status="finished", stage=None, error=None, export=os.path.join(run, "model.ncab"), sizes=sizes, imported=True))
    return run, sizes


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ca.pipeline"); sub = ap.add_subparsers(dest="cmd", required=True)
    def prep_args(p):
        p.add_argument("--start", type=int, default=0, help="first source frame (0-based decoded index)"); p.add_argument("--start-sec", type=float)
        p.add_argument("--frames", type=int, default=24); p.add_argument("--width", type=int, default=32); p.add_argument("--fps", type=float)
        p.add_argument("--threshold", type=int, default=128); p.add_argument("--invert", action="store_true")
    def train_args(p):
        p.add_argument("--preset", choices=list(PRESETS), default="demo"); p.add_argument("--resume", action="store_true")
        for k, t in (("C", int), ("Fh", int), ("steps", int), ("lr", float), ("lamB", float), ("seeds", int)): p.add_argument("--" + k, type=t)
        p.add_argument("--phases", help="phased NCA: uniform:K or auto:K (time-varying CA with an external schedule; ncab-v2 export)")
        p.add_argument("--noise", type=float, help="phased: training-only state noise sigma (0.02 used for the Bad Apple models)")
        p.add_argument("--max-hours", type=float, help="phased: stop cleanly between chunks after this many hours (resumable)")
    p = sub.add_parser("probe"); p.add_argument("video")
    p = sub.add_parser("info"); p.add_argument("export")
    p = sub.add_parser("prep"); p.add_argument("video"); prep_args(p)
    p = sub.add_parser("train"); p.add_argument("run"); p.add_argument("--data", required=True); train_args(p)
    p = sub.add_parser("export"); p.add_argument("run"); p.add_argument("--f16", action="store_true")
    p = sub.add_parser("replay"); p.add_argument("export"); p.add_argument("--no-node", action="store_true")
    p = sub.add_parser("video"); p.add_argument("export"); p.add_argument("--data", required=True); p.add_argument("--out")
    p = sub.add_parser("run"); p.add_argument("video"); p.add_argument("run"); prep_args(p); train_args(p); p.add_argument("--f16", action="store_true")
    p = sub.add_parser("import"); p.add_argument("export"); p.add_argument("run"); p.add_argument("--video", required=True); p.add_argument("--start", type=int, required=True)
    p.add_argument("--width", type=int, required=True); p.add_argument("--threshold", type=int, default=128); p.add_argument("--invert", action="store_true"); p.add_argument("--label"); p.add_argument("--f16", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "probe": print(json.dumps(probe(a.video), indent=1))
        elif a.cmd == "import": print(json.dumps(dict(zip(("run", "sizes"), import_export(a.export, a.run, a.video, a.start, a.width, a.threshold, a.invert, a.label, a.f16)))))
        elif a.cmd == "prep": d, m = prep(a.video, a.start, a.frames, a.width, a.fps, a.threshold, a.invert, a.start_sec); print(json.dumps(dict(data=d, key=m["key"], shape=m["shape"])))
        elif a.cmd == "info": print(json.dumps(info(a.export), indent=1, default=str))
        elif a.cmd == "train": print(json.dumps(train_run(a.run, a.data, a.preset, a.resume, a.phases, a.noise, a.max_hours, C=a.C, Fh=a.Fh, steps=a.steps, lr=a.lr, lamB=a.lamB, seeds=a.seeds)["best"]))
        elif a.cmd == "export": print(json.dumps(dict(zip(("file", "sizes"), export_run(a.run, a.f16)))))
        elif a.cmd == "replay": print(json.dumps({k: v for k, v in replay(a.export, not a.no_node)[2].items() if k != "max_abs_dv_per_frame"}))
        elif a.cmd == "video": print(json.dumps([c["ok"] for c in video(a.export, a.data, a.out or os.path.splitext(a.export)[0] + "_compare")]))
        elif a.cmd == "run":
            os.makedirs(a.run, exist_ok=True); st = os.path.join(a.run, "status.json"); S = dict(status="running", stage=None, error=None, stages={}, pid=os.getpid())
            def stage(name, fn):
                S["stage"] = name; _atomic_json(st, S); t0 = time.time(); r = fn(); S["stages"][name] = dict(seconds=round(time.time() - t0, 1)); _atomic_json(st, S); return r
            try:
                d, m = stage("prep", lambda: prep(a.video, a.start, a.frames, a.width, a.fps, a.threshold, a.invert, a.start_sec))
                stage("train", lambda: train_run(a.run, d, a.preset, a.resume, a.phases, a.noise, a.max_hours, C=a.C, Fh=a.Fh, steps=a.steps, lr=a.lr, lamB=a.lamB, seeds=a.seeds))
                out, sizes = stage("export", lambda: export_run(a.run, a.f16))
                rep = stage("replay", lambda: replay(out)[2]); S["replay"] = {k: v for k, v in rep.items() if k != "max_abs_dv_per_frame"}
                stage("video", lambda: video(out, d, os.path.join(a.run, "compare")))
                S.update(status="finished", stage=None, export=out, sizes=sizes); _atomic_json(st, S); print(json.dumps(S))
            except Exception as e:
                S.update(status="failed", error=f"{type(e).__name__}: {e}"); _atomic_json(st, S); raise
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr); sys.exit(2)


if __name__ == "__main__":
    sys.path.insert(0, ROOT); main()
