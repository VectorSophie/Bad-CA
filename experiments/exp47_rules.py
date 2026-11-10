"""Exp47 (Campaign 12): explicit rule pilot (CPU). Every result is an AUTONOMOUS execution of the learned rule-v1 program
(first frame [+ h0], nothing injected), scored on the full interval; the teacher-forced one-step match is reported
separately.
  direct : k=0, S=1; training pairs = consecutive TRUE frames (teacher-forced); one program, or one per phase (--K)
  extract: from a trained NCA export: run the NCA substep by substep from the first frame, quantise each state
           (v = visible > 0.5; k hidden bits = the k hidden channels with the largest variance, bit = value > that channel's
           median over the run), learn quantised state_i -> state_{i+1} (S=4 substeps per frame), h0 = quantised initial state.
usage: exp47_rules.py OUT_JSON START:LEN MODE [--leaves 16,64,256] [--k 2] [--K 1] [--nca EXPORT]"""
import json, os, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import nca, rules, segment
from ca.diag10 import full_report
from exp45_c12 import target

if __name__ == "__main__":
    out, spec, mode = sys.argv[1], sys.argv[2], sys.argv[3]; opt = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    F = target(spec); T, H, W = F.shape; leaves = [int(x) for x in opt("--leaves", "16,64,256").split(",")]; K = int(opt("--K", 1)); res = []
    phases = [dict(start=1, end=T)] if K == 1 else segment.best_k(F, K, lmin=min(8, (T - 1) // K))["phases"]
    if mode == "extract":
        k = int(opt("--k", 2)); ex, P, f0 = nca.load(opt("--nca", None)); S = ex["S"]; C = P["Wu"].shape[0]
        s = np.zeros((C, H, W), np.float32); s[0] = f0; states = [s.copy()]
        for t in range(1, T):
            for _ in range(S): s = nca.step_np(P, s); states.append(s.copy())
        St = np.stack(states); var = St[:, 1:].var((0, 2, 3)); ch = 1 + np.argsort(-var)[:k]; med = np.median(St[:, ch], (0, 2, 3))
        Vq = (St[:, 0] > 0.5).astype(np.uint8); Hq = (St[:, ch] > med[None, :, None, None]).astype(np.uint8)
        nca_bin = Vq[::S]; frame_of = lambda i: i // S
    else:
        k, S = 0, 1; Vq = F.astype(np.uint8); Hq = np.zeros((T, 0, H, W), np.uint8)
    for L in leaves:
        t0 = time.time(); progs = []; onestep = []
        for p in phases:   # substep index range of frames start..end-1 (pairs (i -> i+1) whose target lies in the phase)
            a, b = (p["start"] - 1) * S, (p["end"] - 1) * S
            X, y, names = rules.dataset(Vq[a:b + 1], Hq[a:b + 1], k)
            cur = X[:, 0].astype(np.int64)
            for b in range(k): cur |= X[:, 1 + b].astype(np.int64) << (b + 1)
            chg = y != cur; bal = float(opt("--balance", 0))   # 0: plain; 1: changing transitions weigh as much in total as keep transitions
            wts = np.where(chg, max(1.0, bal * (~chg).sum() / max(chg.sum(), 1)), 1.0) if bal else None
            lv = rules.fit_tree(X, y, names, max_leaves=L, min_leaf=5, w=wts); prog = rules.tree_program(lv, k); progs.append(prog)
            onestep.append(float(sum(n * pur for _, _, n, pur in lv) / len(y)))
        model = dict(k=k, S=S, T=T, programs=progs, schedule=[dict(start=p["start"], end=p["end"], program=i) for i, p in enumerate(phases)],
                     first=F[0], h0=Hq[0] if k else None)
        B, fires = rules.run(model); r = full_report(B, None, F); fit_s = time.time() - t0
        path = os.path.join(ROOT, "runs", "c12", "rules", f"{spec.replace(':', '_')}_{mode}_k{k}_K{K}_L{L}_b{opt('--balance', 0)}.rule.json"); os.makedirs(os.path.dirname(path), exist_ok=True)
        rules.export(path, model, meta=dict(spec=spec, mode=mode, leaves=L, K=K, nca=opt("--nca", None), note="executable explicit rules; evaluated by autonomous execution"))
        row = dict(spec=spec, mode=mode, k=k, S=S, K=K, max_leaves=L, balance=float(opt("--balance", 0)), file=os.path.relpath(path, ROOT), complexity=rules.complexity(model), onestep_train_match=onestep,
                   err=r["all"]["err"], thirds=[x["err"] for x in r["thirds"]], hold_err=r["hold"], lag1_f1=r["lag1"]["f1_tol"], lag1_mass=r["lag1"]["mass_ratio"],
                   thin=r["thin"], blips=r["blips_spurious"], specks=r["small_cc"], fit_s=round(fit_s, 1))
        if mode == "extract": row["vs_nca_binary_err"] = float((B != nca_bin).mean()); row["nca_err"] = float((nca_bin[1:] != F[1:]).mean())
        res.append(row); c = row["complexity"]
        print(f"{mode} k{k} K{K} L{L}: err {row['err']:.4f} (hold {row['hold_err']:.4f}) thirds {[round(x, 4) for x in row['thirds']]} lag1 {row['lag1_f1']:.2f} mass {row['lag1_mass']:.2f} | "
              f"rules {c['rules']} conds {c['conditions']} bytes {c['file_bytes']} | 1-step {np.mean(onestep):.4f}" + (f" | vsNCA {row['vs_nca_binary_err']:.4f} (NCA err {row['nca_err']:.4f})" if mode == "extract" else ""), flush=True)
    prev = json.load(open(out)) if os.path.exists(out) else []
    json.dump(prev + res, open(out, "w"), indent=1, default=float)
