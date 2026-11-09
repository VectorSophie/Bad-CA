"""Campaign 08 tables (analysis only): runs/c8/**/*.result.json -> runs/c8/results.json + runs/c8/summary08.md."""
import collections, glob, json, os, statistics as stt, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c8_judge import SEGS, level
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
C8 = os.path.join(ROOT, "runs", "c8")
ARM = {("k1_S2", "pix"): "A", ("k1_S2", "mot"): "B", ("k1s_S4", "pix"): "C", ("k1s_S4", "mot"): "D",
       ("k1s_S2", "pix"): "abl-nbr-pix", ("k1s_S2", "mot"): "abl-nbr", ("k1_S4", "pix"): "abl-S4-pix", ("k1_S4", "mot"): "abl-S4", ("k1_S2", "motcur"): "B+curriculum"}
rows = []
for p in sorted(glob.glob(os.path.join(C8, "**", "*.result.json"), recursive=True)):
    if "/smoke/" in p: continue
    r = json.load(open(p)); rep = r["report"]; lv, fol = level(rep)
    phase = os.path.relpath(os.path.dirname(p), C8)
    rows.append(dict(phase=phase, target=r["target"], arm=ARM.get((r["struct"], r["objective"]), "?"), struct=r["struct"], k=r["k"], S=r["S"],
                     objective=r["objective"], lam=r["lam"], seed=r["seed"], level=lv, followed=fol, exact=rep["exact"], J1=rep["J"],
                     **{f"{s}_{m}": rep[s][m] for s in SEGS + ("all",) for m in ("err", "hold", "f1_tol", "prec_tol", "rec_tol", "change_iou", "f1_tol_on", "f1_tol_off", "change_mass_ratio")},
                     late_period=rep["late_period"], late_frozen=rep["late_frozen"], late_flicker=rep["late_flicker"],
                     active_gates=r["active_gates"], rule_bits=r["rule_bits"], file_bytes=r["export"]["file_bytes"],
                     best_baseline_bytes=(r["baselines_bytes"] or {}).get("best"), evals=r["evals"], evals_per_cpu_s=r["evals_per_cpu_s"],
                     search_cpu_s=r["search_cpu_s"], node_replay_equals_python=r["export"]["node_replay_equals_P"],
                     python_file_replay_equals_sim=r["export"]["python_replay_equals_P"], replay_equals_target=r["export"]["replay_equals_target"],
                     node_decode_ms=r["export"]["node_decode_ms"], export=r["export"]["file"], result=os.path.relpath(p, ROOT)))
json.dump(rows, open(os.path.join(C8, "results.json"), "w"), indent=1, default=float)
md = []
def table(sel, title):
    md.extend([f"\n### {title}\n", "| phase | target | arm | obj (λ) | seed | level | err/hold e·m·l | F1tol e·m·l | prec_tol late | mass late | late period | gates | bytes | evals (/s) | Node=Py |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"])
    for r in sorted(sel, key=lambda r: (r["phase"], r["target"], r["arm"], r["lam"], r["seed"])):
        eh = "·".join(f"{r[s + '_err'] / max(r[s + '_hold'], 1e-9):.2f}" for s in SEGS); f1 = "·".join(f"{r[s + '_f1_tol']:.2f}" for s in SEGS)
        md.append(f"| {r['phase']} | {r['target']} | {r['arm']} | {r['objective']} ({r['lam']:g}) | {r['seed']} | {r['level']}{' (exact)' if r['exact'] else ''} | {eh} | {f1} | "
                  f"{r['late_prec_tol']:.2f} | {r['late_change_mass_ratio']:.2f} | {r['late_period']} | {r['active_gates']} | {r['file_bytes']} | {r['evals']} ({r['evals_per_cpu_s']}) | {r['node_replay_equals_python']} |")
for ph in sorted({r["phase"] for r in rows}): table([r for r in rows if r["phase"] == ph], ph)
md.append("\n### Arm summary (per phase/target)\n")
md.append("| phase | target | arm | seeds | success | partial | fail | median late F1tol | late F1tol range | median err/hold (all) | median gates | median evals |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
g = collections.defaultdict(list)
for r in rows: g[(r["phase"], r["target"], r["arm"], r["lam"])].append(r)
for (ph, t, a, lam), rs in sorted(g.items()):
    c = collections.Counter(r["level"] for r in rs); lf = [r["late_f1_tol"] for r in rs]
    md.append(f"| {ph} | {t} | {a} (λ={lam:g}) | {len(rs)} | {c['success']} | {c['partial']} | {c['fail']} | {stt.median(lf):.3f} | {min(lf):.3f}–{max(lf):.3f} | "
              f"{stt.median(r['all_err'] / r['all_hold'] for r in rs):.3f} | {stt.median(r['active_gates'] for r in rs)} | {stt.median(r['evals'] for r in rs)} |")
led = [json.loads(l) for l in open(os.path.join(C8, "cpu_ledger.jsonl"))]; cc = collections.Counter(); ww = collections.Counter()
for r in led: cc[r["kind"]] += r["cpu_s"]; ww[r["kind"]] += r["wall_s"]
md.append("\n### CPU ledger (core-s / wall-s)\n"); md += [f"- {k}: {cc[k]:.0f} / {ww[k]:.0f}" for k in sorted(cc)] + [f"- **total {sum(cc.values()) / 3600:.2f} core-h**; GPU 0"]
open(os.path.join(C8, "summary08.md"), "w").write("\n".join(md) + "\n"); print("\n".join(md))
