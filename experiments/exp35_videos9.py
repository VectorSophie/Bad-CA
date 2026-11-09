"""Exp35 (Campaign 09): comparison videos. All CA frames come from runs/c9/replays/*.npz, produced by exp34 from rules
LOADED FROM THEIR EXPORT FILES; C8 frames are decoded from the C8 export. Originals are for display/diff only.
Selection (docs/CAMPAIGN_09.md): per task and family, the (capacity, regime) with most C passes, then highest median
C lag-4 F1; its median seed is the representative; best and worst seeds are also shown."""
import json, os, statistics as stt, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import container, nca
from ca.video_out import diff_rgb, make_video
from c7_common import target

C9 = os.path.join(ROOT, "runs", "c9"); OUT = os.path.join(C9, "media"); os.makedirs(OUT, exist_ok=True)
rows = json.load(open(os.path.join(C9, "results.json")))
TASKS = {"main": ("ba48_4500_60", 4500, 40), "confirm": ("ba48_1800_60", 1800, 44)}
C8 = {x["name"]: x for x in json.load(open(os.path.join(ROOT, "runs", "c8", "media", "video_index.json"))) if "selection" in x["name"]}
index = []
gray = lambda p: np.repeat((np.clip(p, 0, 1) * 255).astype(np.uint8)[..., None], 3, -1)


def pick(task, fam):
    rs = [r for r in rows if r["task"] == task and r["family"] == fam]; g = {}
    for r in rs: g.setdefault((r["capacity"], r["regime"]), []).append(r)
    key = lambda k: (sum(r["report"]["pass"]["C"] for r in g[k]), stt.median(r["report"]["C"]["lag4"]["f1_tol"] for r in g[k]))
    k = max(g, key=key); s = sorted(g[k], key=lambda r: r["report"]["C"]["lag4"]["f1_tol"])
    return k, s[(len(s) - 1) // 2], s[-1], s[0]


def lab(r):
    if r["family"].startswith("NCA"):
        c, f = (int(x.split("=")[1]) for x in r["capacity"].split()); n = r["size"].split()[0]
        return f"{'d' if 'distilled' in r['family'] else ''}NCA {c}x{f} {r['regime']}-tr s{r['seed']} ({n}p)"
    return f"CGP {r['capacity']} {r['regime']}-tr s{r['seed']} ({r['active_gates']}g)"
def info(r): return f"hidden: {str(int(r['capacity'].split('=')[1].split()[0]) - 1) + ' cont. ch' if r['family'].startswith('NCA') else 'k=1 bit'}, S={r['S']}, hidden init 0"
def line(r, st): R = r["report"][st]; return f"{st}: lag4 F1 {R['lag4']['f1_tol']:.2f} (P {R['lag4']['prec_tol']:.2f}) err/hold {R['err_over_hold']:.2f}"


def vid(name, panels, footer, start, t_off=0):
    checks = make_video(panels, footer, lambda t: f"src frame {start + t}  t={(t_off + t) / 30:5.2f}s", os.path.join(OUT, name))
    index.append(dict(name=name, ok=all(c["ok"] for c in checks), footer=footer))


for task, (clip, s0, t0) in TASKS.items():
    full = target(clip); F = full[t0:t0 + 16]
    sel = {fam: pick(task, fam) for fam in ("CGP", "NCA")}
    R = {fam: {role: dict(np.load(os.path.join(C9, "replays", r["id"] + ".npz"))) for role, r in zip(("rep", "best", "worst"), sel[fam][1:])} for fam in sel}
    rep8 = json.load(open(os.path.join(ROOT, C8[("main_" if task == "main" else "second_") + clip + "_selection"]["representative"])))
    P8 = container.decode(open(os.path.join(ROOT, rep8["export"]["file"]), "rb").read())[1][t0:t0 + 16]
    cg, nc = sel["CGP"][1], sel["NCA"][1]
    vid(f"{task}_1_orig_c8_cgp_nca_autonomousC", [("original", F), (f"C8 CGP s{rep8['seed']} (11 gates k1 S2)", P8), (lab(cg), R["CGP"]["rep"]["C"]), (lab(nc), R["NCA"]["rep"]["C"])],
        [f"{clip} frames {s0 + t0}-{s0 + t0 + 15}: ALL panels autonomous from clip frame {s0} (stage C), no injection, no reset",
         "CGP: " + line(cg, "C") + " | NCA: " + line(nc, "C"), f"NCA: {info(nc)}; CGP: {info(cg)}"], s0 + t0, t0)
    for fam, r in (("NCA", nc), ("CGP", cg)):
        d = R[fam]["rep"]
        panels = [("original", F), (f"A: teacher-forced (true prev frame) {fam}", d["A"]), (f"C: autonomous from frame {s0}", d["C"])]
        if fam == "NCA": panels.append(("C: continuous visible (not binary)", gray(d["Cp"])))
        vid(f"{task}_2_{fam}_orig_tf_autonomous", panels, [lab(r) + " | " + info(r), "A: visible reset to the TRUE previous frame every frame (hidden carried): NOT autonomous",
            f"A tfF1 {r['report']['A']['tf_change_f1']:.2f} err/hold {r['report']['A']['err_over_hold']:.2f} | " + line(r, "C")], s0 + t0, t0)
        for role in ("rep", "best", "worst"):
            rr = sel[fam][1:][("rep", "best", "worst").index(role)]; d = R[fam][role]
            vid(f"{task}_3_{fam}_{role}_orig_C_diff", [("original", F), (lab(rr), d["C"]), ("diff (red FN, blue FP)", diff_rgb(d["C"], F))],
                [f"stage C (autonomous from frame {s0}); {role} seed of {fam} {sel[fam][0]}", line(rr, "C") + " | " + line(rr, "B") + f" | pass C: {rr['report']['pass']['C']}"], s0 + t0, t0)
    index.append(dict(name=f"{task}_selection", CGP=dict(group=sel["CGP"][0], rep=cg["id"], best=sel["CGP"][2]["id"], worst=sel["CGP"][3]["id"]),
                      NCA=dict(group=sel["NCA"][0], rep=nc["id"], best=sel["NCA"][2]["id"], worst=sel["NCA"][3]["id"])))

# reduction: teacher (NCA representative) vs distilled small NCA, stage C
for task, (clip, s0, t0) in TASKS.items():
    if not any(r["family"] == "NCA-distilled" and r["task"] == task for r in rows): continue
    full = target(clip); F = full[t0:t0 + 16]; _, tr, _, _ = pick(task, "NCA"); g, dr, db, dw = pick(task, "NCA-distilled")
    T_ = dict(np.load(os.path.join(C9, "replays", tr["id"] + ".npz"))); Dd = dict(np.load(os.path.join(C9, "replays", dr["id"] + ".npz")))
    vid(f"{task}_4_reduction_teacher_vs_distilled", [("original", F), (lab(tr), T_["C"]), (lab(dr), Dd["C"]), ("diff distilled vs original", diff_rgb(Dd["C"], F))],
        [f"stage C from frame {s0}; distilled model trained on the teacher's own replay (not the original)",
         "teacher " + line(tr, "C") + " | distilled " + line(dr, "C") + f" | distilled pass C {dr['report']['pass']['C']}"], s0 + t0, t0)

# NCA positive control: export the seed-0 checkpoint, replay it from the export, show original / recovered / diff
f0 = np.zeros((24, 80), np.uint8); f0[3:9, 2:8] = 1; f0[11:14, 1:10] = 1; f0[16:22, 3:6] = 1
Fc, _ = nca.rollout_np(nca.known_mover(), f0[None].repeat(30, 0), 30, 4)
P = dict(np.load(os.path.join(C9, "nca", "nca_control_C", "best_seed0.npz"))); ex = os.path.join(C9, "exports", "nca_control_s0.nca.json")
nca.export(ex, P, Fc[0], 30, 4, dict(task="control", seed=0)); _, Pl, _ = nca.load(ex); Vc, _ = nca.rollout_np(Pl, Fc, 30, 4)
vid("control_nca_mover_orig_recovered", [("original (known NCA)", Fc), ("recovered NCA s0 (from export)", Vc), ("diff", diff_rgb(Vc, Fc))],
    [f"NCA positive control: trained from random init on these frames only; exact: {bool((Vc == Fc).all())}", "CGP controls: runs/c8/media/control_* (reused)"], 0)
json.dump(index, open(os.path.join(OUT, "video_index.json"), "w"), indent=1, default=str)
print(json.dumps([(x["name"], x.get("ok")) for x in index]))
