"""Exp34 (Campaign 09): A / A-reset / B / B-warm / C diagnostics for every CGP circuit and NCA seed, pre-registered pass
rules (docs/CAMPAIGN_09.md), exports + independent replays. Rules are LOADED FROM THEIR EXPORT FILES (CGP container
files; NCA nca-v1 JSON written here from the saved checkpoint). Writes runs/c9/results.json, runs/c9/summary09.md and
per-model stage replays runs/c9/replays/<id>.npz (binary frames + NCA continuous visible).
Stages: A = visible set to the TRUE previous frame each frame, hidden carried (starts 0 at the window start);
A_reset = same with hidden := 0 each frame; B = true window frame once, hidden 0, autonomous; B_warm = hidden from a
teacher-forced warm-up over clip frames 0..t0-1, then autonomous from the true window frame; C = clip frame 0, hidden 0,
autonomous to the window end (metrics on the window frames)."""
import glob, hashlib, json, os, subprocess, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import container, nca
from ca.motion import _dil, _prf
from ca.motion9 import window_report
from ca.rule import family_taps
from c7_common import target
from c8_targets import STRUCT

TASKS = {"main": ("ba48_4500_60", 40, 0.50), "confirm": ("ba48_1800_60", 44, 0.65)}
C9 = os.path.join(ROOT, "runs", "c9"); os.makedirs(os.path.join(C9, "replays"), exist_ok=True); os.makedirs(os.path.join(C9, "exports"), exist_ok=True)


class CGP:
    def __init__(self, path):
        _, W, H, T, f0, r = container.decode_rule(open(path, "rb").read())
        self.gates, self.outs, self.taps, self.S, self.C = r["gates"], r["outputs"], r["taps"], r["S"], r["k"] + 1
        n = len(self.taps); need, stack = set(), [o for o in self.outs if o >= n]
        while stack:
            p = stack.pop()
            if p in need: continue
            need.add(p); _, a, b = self.gates[p - n]; stack += [x for x in (a, b) if x >= n]
        self.order, self.n = sorted(need), n
        self.params = f"{len(self.order)} active gates"; self.kind = "CGP circuit (binary)"

    def init(self, f): s = np.zeros((self.C,) + f.shape, np.uint8); s[0] = f; return s
    def vis(self, s): return s[0].astype(np.uint8)
    def setvis(self, s, f): s = s.copy(); s[0] = f; return s
    def reset(self, s): s = s.copy(); s[1:] = 0; return s
    def step(self, s):
        H, W = s.shape[1:]; pd = np.pad(s, ((0, 0), (1, 1), (1, 1)))
        pool = {i: pd[c, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W] for i, (c, dy, dx) in enumerate(self.taps)}
        for p in self.order:
            tt, a, b = self.gates[p - self.n]; pool[p] = (tt >> (pool[a] * 2 + pool[b])) & 1
        return np.stack([pool[o].astype(np.uint8) for o in self.outs])


class NCA:
    def __init__(self, path):
        self.ex, self.P, _ = nca.load(path); self.S, self.C = self.ex["S"], self.ex["C"]
        self.params = f"{self.ex['n_params']} float32 params (C={self.C}, Fh={self.ex['Fh']})"; self.kind = "Neural CA (continuous state)"
    def init(self, f): s = np.zeros((self.C,) + f.shape, np.float32); s[0] = f; return s
    def vis(self, s): return (s[0] > 0.5).astype(np.uint8)
    def setvis(self, s, f): s = s.copy(); s[0] = f; return s
    def reset(self, s): s = s.copy(); s[1:] = 0; return s
    def step(self, s): return nca.step_np(self.P, s)


def run(M, s, frames, force=None, reset=False):
    """Advance len(frames)-1 observed frames; force = list of true frames to set visible before each frame."""
    out, prob = [M.vis(s)], [s[0].astype(np.float32)]
    for t in range(1, frames):
        if force is not None:
            s = M.setvis(s, force[t - 1]); s = M.reset(s) if reset else s
        for _ in range(M.S): s = M.step(s)
        out.append(M.vis(s)); prob.append(s[0].astype(np.float32))
    return np.stack(out), np.stack(prob), s


def tf_report(Pred, F):
    """Pred[t] predicted from TRUE F[t-1] (t = 1..T-1)."""
    P = Pred[1:].astype(bool); prev = F[:-1].astype(bool); cur = F[1:].astype(bool)
    Pon, Poff, Ton, Toff = P & ~prev, ~P & prev, cur & ~prev, ~cur & prev
    hp = (Pon & _dil(Ton)).sum() + (Poff & _dil(Toff)).sum(); ht = (Ton & _dil(Pon)).sum() + (Toff & _dil(Poff)).sum()
    p, r, f = _prf(hp, int(Pon.sum() + Poff.sum()), ht, int(Ton.sum() + Toff.sum()))
    err = float((P != cur).mean()); hold = float((cur != F[:1].astype(bool)).mean())
    return dict(err=err, err_over_hold=err / hold, tf_change_prec=p, tf_change_rec=r, tf_change_f1=f)


def pass_auto(r, thr):
    return r["err_over_hold"] is not None and r["err_over_hold"] <= 0.70 and r["lag4"]["f1_tol"] >= thr and r["lag4"]["prec_tol"] >= 0.40


def evaluate(M, task):
    clip, t0, thr = TASKS[task]; full = target(clip); F = full[t0:t0 + 16]
    s = M.init(F[0]); A, Ap, _ = run(M, s, 16, force=F)
    Ar, _, _ = run(M, M.init(F[0]), 16, force=F, reset=True)
    B, Bp, _ = run(M, M.init(F[0]), 16)
    _, _, sw = run(M, M.init(full[0]), t0 + 1, force=full)               # TF warm-up through frame t0 (visible forced up to t0-1)
    Bw, _, _ = run(M, M.setvis(sw, F[0]), 16)
    Cf, Cp, _ = run(M, M.init(full[0]), t0 + 16)
    rep = {"A": tf_report(A, F), "A_reset": tf_report(Ar, F), "B": window_report(B, F), "B_warm": window_report(Bw, F), "C": window_report(Cf[t0:], F)}
    for k in ("B", "B_warm", "C"): rep[k].pop("err_curve")
    rep["pass"] = dict(A=rep["A"]["tf_change_f1"] >= 0.5 and rep["A"]["err_over_hold"] <= 0.30,
                       B=pass_auto(rep["B"], thr), B_warm=pass_auto(rep["B_warm"], thr), C=pass_auto(rep["C"], thr))
    return rep, dict(A=A, B=B, Bp=Bp, Bw=Bw, C=Cf[t0:], Cp=Cp[t0:], C_full=Cf)


rows = []
# ---- CGP: container exports written by exp27 ----
for p in sorted(glob.glob(os.path.join(C9, "cgp", "*.result.json"))):
    r = json.load(open(p)); tname = r["target"]; clip = tname.split("@")[0]
    task = "main" if clip == "ba48_4500_60" else "confirm"; regime = "C" if "@0-" in tname else "B"
    M = CGP(os.path.join(ROOT, r["export"]["file"])); rep, fr = evaluate(M, task)
    # the export replays its own training regime: check our stage replay equals it
    _, Dexp, _ = container.decode(open(os.path.join(ROOT, r["export"]["file"]), "rb").read())
    same = bool((Dexp == (fr["C_full"] if regime == "C" else fr["B"])).all())
    mid = f"cgp_{task}_{regime}_G{r['G_max']}_s{r['seed']}"; np.savez_compressed(os.path.join(C9, "replays", mid + ".npz"), **{k: v for k, v in fr.items()})
    rows.append(dict(id=mid, family="CGP", task=task, regime=regime, capacity=f"G={r['G_max']}", size=M.params, active_gates=len(M.order), seed=r["seed"],
                     k=r["k"], S=r["S"], file_bytes=r["export"]["file_bytes"], export=r["export"]["file"], evals=r["evals"], search_cpu_s=r["search_cpu_s"],
                     verify=dict(python_node_byte_match=r["export"]["node_replay_equals_P"], stage_replay_equals_export=same), report=rep))
# ---- NCA: export nca-v1 from best checkpoints, replay with numpy (export) and Node ----
for d in sorted(glob.glob(os.path.join(C9, "nca", "nca_*_*"))):
    parts = os.path.basename(d).split("_"); task, regime = parts[1:3]; suffix = "".join("_" + x for x in parts[3:])
    if task == "control": continue
    clip, t0, _ = TASKS[task]; full = target(clip); Fr = full[t0:t0 + 16] if regime == "B" else full[:t0 + 16]
    for ck in sorted(glob.glob(os.path.join(d, "best_seed*.npz"))):
        seed = int(ck[-5]); P = dict(np.load(ck)); mid = f"nca_{task}_{regime}{suffix}_s{seed}"; meta = json.load(open(ck[:-4] + ".json"))
        ex_path = os.path.join(C9, "exports", mid + ".nca.json")
        nca.export(ex_path, P, Fr[0], len(Fr), 4, dict(task=task, regime=regime, seed=seed, clip=clip, start_frame=(t0 if regime == "B" else 0), ckpt=os.path.relpath(ck, ROOT), train=meta))
        M = NCA(ex_path); rep, fr = evaluate(M, task)
        Vpy, Ppy = nca.rollout_np(M.P, Fr, len(Fr), 4)
        nd_bin = os.path.join(C9, "exports", mid + ".node_vis.f32")
        nd = json.loads(subprocess.run(["node", os.path.join(ROOT, "player", "nca_replay.mjs"), ex_path, nd_bin], capture_output=True, text=True, check=True).stdout)
        Pnode = np.fromfile(nd_bin, np.float32).reshape(len(Fr), *Fr[0].shape); os.remove(nd_bin)
        Vnode = np.concatenate([Fr[:1], (Pnode[1:] > 0.5)]).astype(np.uint8)
        stage = fr["C_full"] if regime == "C" else fr["B"]
        rows.append(dict(id=mid, family="NCA" if not suffix else "NCA-distilled", task=task, regime=regime, capacity=f"C={meta['C']} Fh={meta['Fh']}", size=M.params, seed=seed, S=4, file_bytes=os.path.getsize(ex_path),
                         export=os.path.relpath(ex_path, ROOT), train=meta,
                         verify=dict(numpy_vs_node_binary_mismatch_px=int((Vpy != Vnode).sum()), numpy_vs_node_max_abs_visible=float(np.abs(Ppy[1:] - Pnode[1:]).max()),
                                     near_threshold_values=int((np.abs(Ppy[1:] - 0.5) < 1e-3).sum()), node_sha256=nd["sha256"],
                                     numpy_sha256=hashlib.sha256(np.ascontiguousarray(Vpy, np.uint8).tobytes()).hexdigest(),
                                     stage_replay_equals_export_replay=bool((stage == Vpy).all())), report=rep))
        np.savez_compressed(os.path.join(C9, "replays", mid + ".npz"), **{k: v for k, v in fr.items()})
json.dump(rows, open(os.path.join(C9, "results.json"), "w"), indent=1, default=float)
md = ["| id | size | A tfF1 / err/h | B lag4 F1 (P) / err/h | B_warm lag4 F1 | C lag4 F1 (P) / err/h | C revert (true) | pass A B Bw C | verify |", "|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    R = r["report"]; ps = r["report"]["pass"]
    v = r["verify"]; vs = ("py=node " + str(v["python_node_byte_match"])) if r["family"] == "CGP" else f"bin mismatch {v['numpy_vs_node_binary_mismatch_px']} px, max|d| {v['numpy_vs_node_max_abs_visible']:.1e}"
    md.append(f"| {r['id']} | {r['size']} | {R['A']['tf_change_f1']:.2f} / {R['A']['err_over_hold']:.2f} | {R['B']['lag4']['f1_tol']:.2f} ({R['B']['lag4']['prec_tol']:.2f}) / {R['B']['err_over_hold']:.2f} | "
              f"{R['B_warm']['lag4']['f1_tol']:.2f} | {R['C']['lag4']['f1_tol']:.2f} ({R['C']['lag4']['prec_tol']:.2f}) / {R['C']['err_over_hold']:.2f} | "
              f"{R['C']['revert_pred']} ({R['C']['revert_true']:.2f}) | {'.'.join('Y' if ps[k] else '-' for k in ('A', 'B', 'B_warm', 'C'))} | {vs} |")
open(os.path.join(C9, "summary09.md"), "w").write("\n".join(md) + "\n"); print("\n".join(md))
