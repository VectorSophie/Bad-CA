"""Exp36 (Campaign 10, before any training): full autonomous replay of the Campaign 9 C-trained NCAs from their
EXPORTS ONLY (first frame + weights; the original clip is never read by the replay), check against the C9 evaluation,
diagnose the errors on arrays, and render the pre-training videos.
Writes runs/c10/pre_diag.json and runs/c10/media/pre10_*.  usage: exp36_full10.py"""
import json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import nca
from ca.diag10 import full_report, shift_curve, error_runs, band
from ca.video_out import diff_rgb, make_video
from c7_common import target

C9, C10 = os.path.join(ROOT, "runs", "c9"), os.path.join(ROOT, "runs", "c10"); OUT = os.path.join(C10, "media"); os.makedirs(OUT, exist_ok=True)
TASKS = {"main": ("ba48_4500_60", 4500, 40, 2), "confirm": ("ba48_1800_60", 1800, 44, 1)}   # clip, first src frame, t0, C9 representative seed
CROP = {"main": (4, 32, 0, 28), "confirm": (0, 28, 20, 48)}   # y0, y1, x0, x1 of the zoom (main: arm/shoulder; confirm: staff + hand), fixed by eye before training


def gray(V): return np.repeat((np.clip(V, 0, 1) * 255).astype(np.uint8)[..., None], 3, -1)


def replay_export(path):
    ex, P, f0 = nca.load(path)
    B, V = nca.rollout_np(P, f0[None], ex["n_frames"], ex["S"])               # only f0 is given: no access to later frames
    return ex, P, f0, B, V


def counterfactual(P, F, t0, S, T):
    """Diagnostic: autonomous to t0, then visible := TRUE F[t0] once (hidden kept), autonomous to the end."""
    C = P["Wu"].shape[0]; s = np.zeros((C,) + F[0].shape, np.float32); s[0] = F[0]; out = [F[0]]
    for t in range(1, T):
        if t == t0 + 1: s[0] = F[t0]
        for _ in range(S): s = nca.step_np(P, s)
        out.append((s[0] > 0.5).astype(np.uint8))
    return np.stack(out)


if __name__ == "__main__":
    res = {}
    for task, (clip, s0, t0, rep) in TASKS.items():
        full = target(clip); res[task] = {}
        for seed in range(3):
            ex, P, f0, B, V = replay_export(os.path.join(C9, "exports", f"nca_{task}_C_s{seed}.nca.json")); T = len(B); F = full[:T]
            old = np.load(os.path.join(C9, "replays", f"nca_{task}_C_s{seed}.npz"))
            chk = dict(T=T, src_frames=[s0, s0 + T - 1], first_frame_equals_original=bool((f0 == F[0]).all()),
                       full_equals_c9_C_full=bool((old["C_full"] == B).all()), window_equals_c9_C=bool((old["C"] == B[t0:]).all()),
                       window_cont_max_abs_vs_c9=float(np.abs(old["Cp"] - V[t0:]).max()))
            r = full_report(B, V, F, (t0, t0 + 15)); idx = range(1, T)
            r["check"] = chk
            r["shift_posthoc"] = dict(full=shift_curve(B, F, idx), window=shift_curve(B, F, range(t0, t0 + 16)))
            r["threshold_sweep_diag"] = {str(th): float(((V[1:] > th) != F[1:].astype(bool)).mean()) for th in (0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7)}
            Fb = F[1:].astype(bool)
            r["cont_values"] = dict(q_on=np.quantile(V[1:][Fb], [0.01, 0.05, 0.5, 0.95, 0.99]).round(3).tolist(),
                                    q_off=np.quantile(V[1:][~Fb], [0.01, 0.05, 0.5, 0.95, 0.99]).round(3).tolist(),
                                    frac_lt0=float((V[1:] < 0).mean()), frac_gt1=float((V[1:] > 1).mean()), min=float(V.min()), max=float(V.max()))
            E = B != F.astype(np.uint8); y0, y1, x0, x1 = CROP[task]; w = slice(t0, t0 + 16)
            # accumulated vs new: window error pixel-frames whose error run began before the window start
            Ew = E.copy(); started_before = 0; tot = int(E[t0:t0 + 16].sum())
            run_start = np.full(E.shape[1:], -1)
            for t in range(T):
                run_start = np.where(E[t] & (run_start < 0), t, np.where(E[t], run_start, -1))
                if t0 <= t < t0 + 16: started_before += int((E[t] & (run_start < t0)).sum())
            cf = counterfactual(P, F, t0, ex["S"], T)
            Ecrop = E[:, y0:y1, x0:x1]; bd = band(F)[:, y0:y1, x0:x1]; Fc = F[:, y0:y1, x0:x1].astype(bool); Bc = B[:, y0:y1, x0:x1].astype(bool)
            r["accumulation"] = dict(window_err_pixelframes=tot, frac_started_before_window=started_before / max(tot, 1),
                                     window_err_C=float(E[w][1:].mean()), window_err_visible_corrected_at_t0=float((cf[w] != F[w])[1:].mean()),
                                     err_curve_first_third=float(E[1:T // 3].mean()), err_curve_window=float(E[w].mean()))
            r["crop"] = dict(box=CROP[task], window_err_in_crop_px=int(Ecrop[w].sum()), window_err_outside_crop_px=int(E[w].sum() - Ecrop[w].sum()),
                             fn_runs=np.bincount(error_runs(Fc & ~Bc), minlength=8)[1:].tolist(), fp_runs=np.bincount(error_runs(~Fc & Bc), minlength=8)[1:].tolist(),
                             crop_err_band_frac=float(Ecrop[bd].sum() / max(Ecrop.sum(), 1)))
            res[task][f"s{seed}"] = r
            if seed == rep:
                lab = lambda t, off=0: f"src frame {s0 + off + t}  t={(off + t) / 30:5.2f}s"
                ttl = f"C9 nca_{task}_C_s{seed} (rep., 10,320 params)"
                foot = [f"FULL autonomous replay from the export: src {s0}-{s0 + T - 1} ({T} frames), first frame + hidden 0, no reset/injection",
                        f"diff: red = FN (orig on, replay off), blue = FP; continuous: gray = clip(v,0,1), threshold 0.5, v range [{V.min():.2f}, {V.max():.2f}]",
                        f"full err {r['all']['err']:.4f}  err in contour band {r['all']['frac_err_in_band']:.0%}  lag4 F1 {r['lag4']['f1_tol']:.2f}"]
                pan = lambda sl, ys=slice(None), xs=slice(None): [("original", F[sl][:, ys, xs]), (ttl if ys == slice(None) else "C9 binary", B[sl][:, ys, xs]),
                                                                  ("continuous v (gray)", gray(V[sl][:, ys, xs])), ("diff red FN blue FP", diff_rgb(B[sl][:, ys, xs], F[sl][:, ys, xs]))]
                make_video(pan(slice(None)), foot, lab, os.path.join(OUT, f"pre10_{task}_FULL_{s0}-{s0 + T - 1}_orig_c9_cont_diff"))
                make_video(pan(w), [f"WINDOW src {s0 + t0}-{s0 + t0 + 15} CUT FROM the full autonomous replay above (not restarted)"] + foot[1:], lambda t: lab(t, t0),
                           os.path.join(OUT, f"pre10_{task}_WINDOW_{s0 + t0}-{s0 + t0 + 15}_orig_c9_cont_diff"))
                make_video(pan(slice(None), slice(y0, y1), slice(x0, x1)), [f"ZOOM rows {y0}-{y1 - 1} cols {x0}-{x1 - 1} of the FULL replay, nearest-neighbour x12"] + foot[:2], lab,
                           os.path.join(OUT, f"pre10_{task}_ZOOM_FULL_{s0}-{s0 + T - 1}_orig_c9_cont_diff"), scale=12)
        make_video([("original", F)] + [(f"C9 s{s}", replay_export(os.path.join(C9, "exports", f"nca_{task}_C_s{s}.nca.json"))[3]) for s in range(3)],
                   [f"FULL autonomous replays of all three C9 seeds from their exports, src {s0}-{s0 + T - 1}"], lambda t: f"src frame {s0 + t}  t={t / 30:5.2f}s",
                   os.path.join(OUT, f"pre10_{task}_FULL_{s0}-{s0 + T - 1}_orig_c9_s0_s1_s2"))
    json.dump(res, open(os.path.join(C10, "pre_diag.json"), "w"), indent=1)
    for task in res:
        for s, r in res[task].items():
            a = r["all"]; print(task, s, r["check"], f"err {a['err']:.4f} fp {a['fp']:.4f} fn {a['fn']:.4f} band {a['frac_err_in_band']:.2f}",
                                f"margins<.1 {a['err_margin_lt_0_1']:.2f} .1-.25 {a['err_margin_0_1_to_0_25']:.2f} >=.25 {a['err_margin_ge_0_25']:.2f}",
                                "thirds", [round(x['err'], 4) for x in r["thirds"]], "runs", r["err_runs"], "blips", r["blips_spurious"], r["blips_true"],
                                "cc", r["small_cc"], "shift", r["shift_posthoc"], "acc", r["accumulation"], "crop", r["crop"], "thr", r["threshold_sweep_diag"], "cont", r["cont_values"], sep="\n  ")
