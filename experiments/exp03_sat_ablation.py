"""Exp03: state-representation ablation by exact SAT synthesis (CPU only, never via gpuq).

Family F_priv(k): inputs = visible Moore 3x3 (bits 0..8) + the cell's own k hidden bits;
outputs = next visible + next k hidden; one LUT per output channel, shared by all cells.
Boundary: constant 0 for every channel. Hidden init: all 0 (declared, 0 seed bits).
Observed channel: visible only, compared after every `S` generations; target cadence = every
frame of the clip (stride 1). Success = exact reproduction of all frames by autonomous rollout.

  A k=0 S=1 | B k=1 S=1 | C k=2 S=1 | D k=0 S=2 | E k=1 S=2
  extras: X1 k=2 S=2 | X2 k=1 S=3 | X3 k=0 S=3 | X4 F_shared(1) S=1 | X5 k=0 S=4 | X6 k=1 S=4

usage: exp03_sat_ablation.py run [--wall S] [--cpu S] [--mem GB] [--jobs N] [--conds A,B,..] [--phase none|keep] [targets...]
       exp03_sat_ablation.py worker TARGET COND [PHASE]
Statuses: SAT, UNSAT (solver proof), UNKNOWN_timeout, UNKNOWN_memout, ERROR (implementation).
"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "runs", "exp03")
CONDS = {"A": (0, 1, "self"), "B": (1, 1, "self"), "C": (2, 1, "self"), "D": (0, 2, "self"), "E": (1, 2, "self"),
         # extras beyond the requested ablation, labelled X*:
         "X1": (2, 2, "self"), "X2": (1, 3, "self"), "X3": (0, 3, "self"),
         "X4": (1, 1, "moore"),  # F_shared(1): neighbours also read the hidden bit
         "X5": (0, 4, "self"), "X6": (1, 4, "self")}


def target(name):
    from ca.synthetic import TINY
    if name in TINY:
        return TINY[name]()
    if name.startswith("ba"):  # ba<width>_<startframe>_<len>  (explicit non-uniform start, skips black opening)
        from ca.video import binarize, gray_frames
        w, s, l = map(int, name[2:].split("_"))
        F = binarize(gray_frames(os.path.join(ROOT, "refs", "Bad-Apple-Starter", "BadApple!!.mp4"), w)[0])[s:s + l]
        assert F[0].min() != F[0].max(), "first frame must be non-uniform"
        return F
    raise KeyError(name)


def worker(tname, cond, phase="none"):
    """phase='keep': solver decision phases prefer LUT outputs that copy the cell's own current
    value (visible->centre bit, hidden->own hidden bit): a cheap bias toward simpler rules."""
    from pysat.solvers import Cadical195
    from ca.rule import family_taps, rollout
    from ca.sat import encode, history_conflicts, light_cone_conflicts, luts_from_model, state_repeat_conflict
    from ca.export import export_rule
    k, S, seen_by = CONDS[cond]
    F = target(tname); T = len(F)
    taps = family_taps(k, seen_by)
    res = dict(target=tname, cond=cond, hidden_bits=k, steps_per_frame=S,
               family="F_priv" if seen_by == "self" else "F_shared", boundary=0,
               hidden_init=0, shape=list(F.shape), solver="cadical195 (pysat), default options, deterministic",
               phase=phase)
    stem = f"{tname}_{cond}" + ("" if phase == "none" else f"_{phase}")
    if k == 0:
        res["state_repeat_proof"] = state_repeat_conflict(F)
    if S == 1 and seen_by == "self":
        res["history_conflicts_any_k"] = history_conflicts(F)
    res["light_cone_conflicts"] = light_cone_conflicts(F, S)
    t0 = time.perf_counter()
    clauses, L, nvars = encode(F, taps, k + 1, S)
    res.update(encode_s=round(time.perf_counter() - t0, 2), n_vars=nvars, n_clauses=len(clauses), lut_vars=len(L))
    print(json.dumps(dict(progress="encoded", **res)), flush=True)
    with Cadical195(bootstrap_with=clauses) as s:
        del clauses
        if phase == "keep":
            own = {0: 4}; own.update({c: taps.index((c, 0, 0)) for c in range(1, k + 1)})
            s.set_phases([v if (e >> own[c]) & 1 else -v for (c, e), v in L.items()])
        t0 = time.perf_counter(); ok = s.solve(); res["solve_s"] = round(time.perf_counter() - t0, 2)
        res["status"] = "SAT" if ok else "UNSAT"
        if ok:
            luts = luts_from_model(s.get_model(), L, k + 1, len(taps))
    if ok:
        vis, _, seen = rollout(F[0], taps, luts, T, S)
        res["python_rollout_exact"] = bool((vis == F).all())
        res["visited_entries"] = len(seen)
        path = os.path.join(OUT, f"{stem}.rule.json")
        np.save(os.path.join(OUT, f"{stem}.luts.npy"), luts)
        json.dump(seen, open(os.path.join(OUT, f"{stem}.visited.json"), "w"))
        ex = export_rule(path, taps, luts, F[0], T, S, meta=dict(source=tname, method="SAT", cond=cond), expected=F)
        binp = path.replace(".rule.json", ".node.bin")
        node = json.loads(subprocess.run(["node", os.path.join(ROOT, "player", "replay.mjs"), path, binp],
                                         capture_output=True, text=True, check=True).stdout)
        res["node_hash_matches_target"] = node["sha256"] == ex["expected_sha256"]
        res["node_frames_equal_target"] = bool((np.fromfile(binp, np.uint8).reshape(F.shape) == F).all())
        res["sizes"] = dict(lut_bits_full=int(luts.size), specified_entries=len(seen),
                            specified_bits=len(seen) * (k + 1), hidden_bits_per_cell=k,
                            hidden_state_bits_total=k * F[0].size, seed_visible_bits=int(F[0].size), hidden_init_bits=0)
    print(json.dumps(dict(progress="done", **res)), flush=True)


def run_one(tname, cond, wall, cpu, mem_gb, phase="none"):
    stem = f"{tname}_{cond}" + ("" if phase == "none" else f"_{phase}")
    log = os.path.join(OUT, f"{stem}.log")
    cmd = ["prlimit", f"--as={mem_gb << 30}", f"--cpu={cpu}", "timeout", "--signal=KILL", str(wall),
           sys.executable, os.path.abspath(__file__), "worker", tname, cond, phase]
    t0 = time.perf_counter()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    wall_s = time.perf_counter() - t0
    lines = [json.loads(l) for l in open(log) if l.startswith("{")]
    res = dict(lines[-1]) if lines else dict(target=tname, cond=cond)
    res.pop("progress", None)
    text = open(log).read()
    if rc == 0 and "status" in res:
        pass
    elif rc in (137, -9) or rc in (152, -24) or "SIGXCPU" in text:
        res["status"] = "UNKNOWN_timeout"  # wall KILL (137) or CPU limit (SIGXCPU)
    elif "MemoryError" in text or "std::bad_alloc" in text or "Cannot allocate" in text:
        res["status"] = "UNKNOWN_memout"
    else:
        res["status"] = "ERROR"; res["error_tail"] = text[-1500:]
    res.update(returncode=rc, wall_s=round(wall_s, 1), limits=dict(wall_s=wall, cpu_s=cpu, mem_gb=mem_gb))
    json.dump(res, open(os.path.join(OUT, f"{stem}.result.json"), "w"), indent=1)
    print(f"{stem:20s} {cond}  {res['status']:16s} wall={wall_s:7.1f}s vars={res.get('n_vars')} clauses={res.get('n_clauses')}", flush=True)
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "worker":
        worker(*sys.argv[2:5]); sys.exit(0)
    args = sys.argv[2:]
    opt = {"--wall": 1800, "--cpu": 1800, "--mem": 16, "--jobs": 10, "--conds": "A,B,C,D,E", "--phase": "none"}
    names = []
    i = 0
    while i < len(args):
        if args[i] in opt: opt[args[i]] = type(opt[args[i]])(args[i + 1]); i += 2
        else: names.append(args[i]); i += 1
    names = names or ["disc12", "bar12"]
    jobs = [(t, c) for t in names for c in opt["--conds"].split(",")]
    with ThreadPoolExecutor(opt["--jobs"]) as ex:
        list(ex.map(lambda tc: run_one(tc[0], tc[1], opt["--wall"], opt["--cpu"], opt["--mem"], opt["--phase"]), jobs))
