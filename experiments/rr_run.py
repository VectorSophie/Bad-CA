"""Readable Rules driver: fit dlrule programs by rollout search ('library' = extend simple classic rules, 'tree' = direct
search started from a decision list fitted to the frames), export them, replay the EXPORT and score it.
TASK: control:life | control:shift | control:blink | control:excitable | START:LEN (Bad Apple, 48x36, threshold 128)
usage: rr_run.py TASK OUTDIR --start library|tree [--k 0] [--S 1] [--K 1] [--seg uniform|auto] [--evals 2000] [--gamma 0]
                 [--seeds 0,1,2] [--max-rules 8] [--max-conds 3] [--teacher EXPORT]
Writes runs/readable_rules/<OUTDIR>/s<seed>.rule.json, .txt (readable), result.json (metrics per seed, complexity, cost)."""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import rules, rr_search, segment
from ca.diag10 import full_report


def control(name, T=24, H=32, W=32, seed=0):
    """Generator runs for the positive controls. seed 0 = the original fixed initial state; other seeds = random held-out ones."""
    rng = np.random.default_rng(seed); f0 = np.zeros((H, W), np.uint8)
    def blobs(n, lo, hi):
        for _ in range(n): y, x, h_, w_ = rng.integers(0, H - hi), rng.integers(0, W - hi), rng.integers(lo, hi), rng.integers(lo, hi); f0[y:y + h_, x:x + w_] = 1
    if name == "life": f0 = (rng.random((H, W)) < 0.35).astype(np.uint8); prog, k = rr_search.LIBRARY["life_b3s23"], 0
    elif name == "shift":
        if seed == 0: f0[8:14, 3:9] = 1; f0[18:24, 2:6] = 1
        else: blobs(3, 2, 7)
        prog, k = rr_search.LIBRARY["shift_from_l"], 0
    elif name == "blink":   # memory: a hidden clock toggles every frame; visible flips only when the clock is 1 -> period-4 pattern
        if seed == 0: f0[10:20, 10:20] = 1
        else: blobs(3, 3, 10)
        k = 1
        prog = {"rules": [{"if": [["h0", "==", 1], ["v", "==", 1]], "then": {"v": 0, "h": 0}}, {"if": [["h0", "==", 1], ["v", "==", 0]], "then": {"v": 1, "h": 0}},
                          {"if": [["h0", "==", 0], ["v", "==", 1]], "then": {"v": 1, "h": 1}}, {"if": [["h0", "==", 0], ["v", "==", 0]], "then": {"v": 0, "h": 1}}]}
    elif name == "excitable":   # Greenberg-Hastings-like: excited (v=1) -> refractory (h=1) -> rest; rest fires if any excited neighbour
        if seed == 0: f0[15:17, 2:4] = 1
        else: blobs(2, 1, 3)
        k = 1
        prog = {"rules": [{"if": [["v", "==", 1]], "then": {"v": 0, "h": 1}}, {"if": [["h0", "==", 1]], "then": {"v": 0, "h": 0}},
                          {"if": [["n8", ">=", 1]], "then": {"v": 1, "h": 0}}]}
    m = dict(k=k, S=1, T=T, programs=[prog], schedule=[dict(start=1, end=T, program=0)], first=f0, h0=np.zeros((k, H, W), np.uint8) if k else None)
    return rules.run(m)[0], m


def target(task):
    if task.startswith("control:"): return control(task.split(":")[1])[0]
    from exp45_c12 import target as t; return t(task)


def evaluate(B, F, teacherB=None):
    r = full_report(B, None, F); e = (B != F).mean((1, 2)); T = len(F)
    w = max(float(e[a:a + 16].mean()) for a in range(1, max(2, T - 15)))
    out = dict(err=r["all"]["err"], hold_err=float((F[1:] != F[:1]).mean()), thirds=[x["err"] for x in r["thirds"]], worst16=w,
               lag1_f1=r["lag1"]["f1_tol"], lag4_f1=r["lag4"]["f1_tol"], lag4_mass=r["lag4"]["mass_ratio"], thin=r["thin"], specks=[r["small_cc"]["pred_specks"], r["small_cc"]["true_specks"]],
               holes=[r["small_cc"]["pred_holes"], r["small_cc"]["true_holes"]], spurious_blips=r["blips_spurious"], exact=bool((B == F).all()))
    if teacherB is not None: out["teacher_agreement_err"] = float((B[1:] != teacherB[1:]).mean())
    return out


def one(args):
    task, outdir, seed, o = args; F = target(task); T = len(F); rng = np.random.default_rng(seed)
    phases = [dict(start=1, end=T)] if o["K"] == 1 else (segment.uniform_k(T, o["K"]) if o["seg"] == "uniform" else segment.best_k(F, o["K"], lmin=min(8, (T - 1) // o["K"]))["phases"])
    t0 = time.process_time(); w0 = time.time()
    model, recs = rr_search.fit_phases(F, phases, o["k"], o["S"], rng, o["start"], o["evals"], o["gamma"], o["max_rules"], o["max_conds"], log=lambda s: None)
    path = os.path.join(outdir, f"s{seed}.rule.json"); rules.export(path, model, meta=dict(task=task, seed=seed, **o))
    L = rules.load(path); B, fires = rules.run(L)                         # score the EXPORT
    tb = None
    if o.get("teacher") and not task.startswith("control:"):
        from ca import phase_nca
        a0, n = map(int, task.split(":")); hd, E, sched, h0, f0 = phase_nca.load(o["teacher"]); Bt, _ = phase_nca.rollout_np(E, f0, sched[:a0 + n], hd["S"], h0); tb = Bt[a0:a0 + n]
    ev = evaluate(B, F, tb); cx = rules.complexity(L); txt = []
    for i, p in enumerate(rules.resolve(L["programs"])): txt.append(f"# program {i}" + (f" (= program {L['programs'][i]['base']} + {len(L['programs'][i]['edits'])} edits)" if "base" in L["programs"][i] else "") + "\n" + rules.to_text(p, o["k"]))
    open(path[:-10] + ".txt", "w").write("\n\n".join(txt) + "\n")
    return dict(seed=seed, file=os.path.relpath(path, ROOT), eval=ev, complexity=cx, phases=recs, cpu_s=round(time.process_time() - t0, 1), wall_s=round(time.time() - w0, 1))


if __name__ == "__main__":
    task, out = sys.argv[1], os.path.join(ROOT, "runs", "readable_rules", sys.argv[2]); os.makedirs(out, exist_ok=True)
    opt = lambda k, d, c=str: c(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    o = dict(start=opt("--start", "library"), k=opt("--k", 0, int), S=opt("--S", 1, int), K=opt("--K", 1, int), seg=opt("--seg", "uniform"), evals=opt("--evals", 2000, int),
             gamma=opt("--gamma", 0.0, float), max_rules=opt("--max-rules", 8, int), max_conds=opt("--max-conds", 3, int), teacher=opt("--teacher", None))
    seeds = [int(s) for s in opt("--seeds", "0,1,2").split(",")]
    with Pool(len(seeds)) as pool: res = pool.map(one, [(task, out, s, o) for s in seeds])
    json.dump(dict(task=task, options=o, results=res), open(os.path.join(out, "result.json"), "w"), indent=1, default=float)
    for r in res:
        e, c = r["eval"], r["complexity"]
        print(f"{task} s{r['seed']}: err {e['err']:.4f} (hold {e['hold_err']:.4f}) exact {e['exact']} lag4 F1 {e['lag4_f1']:.2f} mass {e['lag4_mass'] or 0:.2f} worst16 {e['worst16']:.3f} "
              f"| programs {c['programs']} rules {c['rules']} conds {c['conditions']} edits {c['edits']} bytes {c['file_bytes']} | cpu {r['cpu_s']} s"
              + (f" | vs teacher {e['teacher_agreement_err']:.4f}" if "teacher_agreement_err" in e else ""), flush=True)
