"""Stage 0 interpretation of the selected m0 codebook (K from summary.json): what each shared rule does, measured.
Per code: the pool windows it explains (true motion statistics), its own autonomous behaviour from every pool window
start (15 and 64 steps), its flip set as a minimal sum of products over the patterns seen in the pool (don't-cares
elsewhere), and pairwise differences between codes (pool-frequency-weighted). Labels follow fixed rules declared here:
  static       flips per pixel per step < 0.001 in the 15-step rollout
  growth       births >= 2 x deaths and foreground area ratio (step 15 / step 0) > 1.03
  contraction  deaths >= 2 x births and area ratio < 0.97
  translation  mean centroid speed >= 0.2 px/step and 0.97 <= area ratio <= 1.03
  deformation  otherwise
usage: s0_interp.py  -> runs/context_s0/modes.json"""
import json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
import s0_codebook as s0
import pd_vis
from ca.minimize import sop

S = json.load(open(os.path.join(s0.OUT, "summary.json"))); K = S["select"]["m0"]; CB = np.load(os.path.join(s0.OUT, "codebooks.npz"))
Ts, asg = CB[f"m0_K{K}_global"], CB[f"asg_m0_K{K}_global"]; man = json.load(open(s0.MAN)); V = s0.video(); pool = man["pool"]
R = json.load(open(os.path.join(s0.OUT, "results.json")))

freq = np.zeros(512, np.int64)
for t in pool:
    k, _ = s0.window_xy(V, t, "m0"); freq += np.bincount(k.ravel(), minlength=512)
seen = np.flatnonzero(freq)


def fg(f): return f != pd_vis.bg_colour(f)
def cen(m): return np.array(np.nonzero(m)).mean(1) if m.any() else None


def behaviour(T, n=15):
    flips, births, deaths, ar, sp, frozen = [], [], [], [], [], []
    for t in pool:
        B = s0.roll(T[None], V[t - 1], V[t], n, "m0"); F = np.concatenate([V[t:t + 1], B])
        d = F[1:].astype(int) - F[:-1]; flips.append((d != 0).mean()); births.append((d > 0).mean()); deaths.append((d < 0).mean())
        m0, m1 = fg(V[t]), fg(F[-1])
        if m0.sum() >= 20: ar.append(m1.sum() / m0.sum())
        c0, c1 = cen(m0), cen(m1)
        if c0 is not None and c1 is not None and m0.sum() >= 20: sp.append(np.linalg.norm(c1 - c0) / n)
        frozen.append(bool((B[-1] == B[-2]).all()))
    return dict(flips=float(np.mean(flips)), births=float(np.mean(births)), deaths=float(np.mean(deaths)), area_ratio=float(np.median(ar)),
                speed=float(np.median(sp)), frozen_frac=float(np.mean(frozen)))


def label(b):
    if b["flips"] < 0.001: return "static"
    if b["births"] >= 2 * b["deaths"] and b["area_ratio"] > 1.03: return "growth"
    if b["deaths"] >= 2 * b["births"] and b["area_ratio"] < 0.97: return "contraction"
    if b["speed"] >= 0.2 and 0.97 <= b["area_ratio"] <= 1.03: return "translation"
    return "deformation"


def true_stats(ws):
    ch, br, de, vel = [], [], [], []
    for t in ws:
        F = V[t:t + s0.L]; d = F[1:].astype(int) - F[:-1]; ch.append((d != 0).mean()); br.append((d > 0).mean()); de.append((d < 0).mean())
        c0, c1 = cen(fg(F[0])), cen(fg(F[-1]))
        if c0 is not None and c1 is not None: vel.append(((c1 - c0) / (s0.L - 1)).tolist())
    return dict(n=len(ws), change=float(np.mean(ch)), births=float(np.mean(br)), deaths=float(np.mean(de)),
                mean_velocity_yx=np.mean(vel, 0).tolist() if vel else None)


NAMES = ["NW", "N", "NE", "W", "C", "E", "SW", "S", "SE"]
modes = []
for c in range(K):
    T = Ts[c]; G = s0.to_flip(T); ws = [t for t, a in zip(pool, asg) if a == c]
    expr = sop(G, seen, 9) if G[seen].any() else None
    d_ = expr[0] if expr is not None else None; e = d_["expr"] if d_ else "False"; nterms = d_["terms"] if d_ else 0
    for i in reversed(range(9)): e = e.replace(f"x{i}", NAMES[i])
    b15 = behaviour(T, 15); b64 = behaviour(T, 64)
    used = sorted({r["e"]["scene"] for r in R["eval"] if r["mode"] == "m0" and r["e"]["split"] == "test" and r[f"K{K}"]["oracle1"]["code"] == c})
    modes.append(dict(code=c, pool_windows=true_stats(ws), behaviour_15=b15, behaviour_64=b64, label_15=label(b15),
                      flip_entries=int(G.sum()), flip_entries_seen=int(G[seen].sum()), flip_weight=float(freq[G.astype(bool)].sum() / freq.sum()),
                      flip_sop=e, sop_terms=nterms, test_scenes_as_oracle=used))
diff = [[float(freq[(s0.to_flip(Ts[a]) != s0.to_flip(Ts[b]))].sum() / freq.sum()) for b in range(K)] for a in range(K)]
ndiff = [[int((s0.to_flip(Ts[a]) != s0.to_flip(Ts[b]))[seen].sum()) for b in range(K)] for a in range(K)]
json.dump(dict(K=K, modes=modes, pair_diff_weighted=diff, pair_diff_patterns=ndiff, seen_patterns=len(seen)), open(os.path.join(s0.OUT, "modes.json"), "w"), indent=1)
for m in modes:
    print(f'code {m["code"]}: {m["pool_windows"]["n"]:3d} pool windows (true change {m["pool_windows"]["change"]:.4f}, b/d {m["pool_windows"]["births"]:.4f}/{m["pool_windows"]["deaths"]:.4f}, v {np.round(m["pool_windows"]["mean_velocity_yx"], 2) if m["pool_windows"]["mean_velocity_yx"] else None}) '
          f'| rollout15 {m["label_15"]} flips {m["behaviour_15"]["flips"]:.4f} b/d {m["behaviour_15"]["births"]:.4f}/{m["behaviour_15"]["deaths"]:.4f} area {m["behaviour_15"]["area_ratio"]:.2f} speed {m["behaviour_15"]["speed"]:.2f} frozen15 {m["behaviour_15"]["frozen_frac"]:.2f} frozen64 {m["behaviour_64"]["frozen_frac"]:.2f} '
          f'| flips: {m["flip_entries_seen"]} patterns, weight {m["flip_weight"]:.4f}, {m["sop_terms"]} terms | test scenes {m["test_scenes_as_oracle"]}')
    print("   flip if:", m["flip_sop"][:400])
print("pairwise differing seen patterns:", ndiff)
print("pairwise differing weight:", np.round(diff, 4).tolist())
