"""Phase 07 tables (analysis only): every result.json of c2/c3/c4/c5 -> runs/c7/results.json + runs/c7/summary07.md.
No combined score: each axis is a column."""
import collections, glob, json, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
C7 = os.path.join(ROOT, "runs", "c7")
CONTROLS = {"life_b3s23", "so_ctrl", "pcA", "pcB", "pcA_long"}
COMMON = {"ba48_4500_60", "ba32_3000_5"}
rows = []
for p in sorted(glob.glob(os.path.join(C7, "c*", "**", "*result.json"), recursive=True)):
    if "c4_smoke" in p: continue
    r = json.load(open(p)); e = r["eval"]; m = r["method"].split()[0]
    if m == "C2": cx = f"{r['best_rule']} ({r['n_conditions']} set bits of 18)"; units = dict(rule_bits=18)
    elif m == "C3": cx = f"tree: {r['internal_nodes']} conditions, {r['leaves']} leaves ({r['circuit_gates']} gates compiled)"; units = dict(conditions=r["internal_nodes"], gates=r["circuit_gates"], rule_native_bits=r["rule_native_bits"])
    elif m == "C4": cx = f"{r['active_gates']} active gates (of G={r['G_max']}), seed {r['seed']}"; units = dict(gates=r["active_gates"], rule_bits=r["rule_bits"])
    else: cx = f"{r['live_gates']} live gates, seed {r['selected_seed']} of 4"; units = dict(gates=r["live_gates"], rule_bits=r["rule_bits"])
    S = r.get("S") or {"k1_S1": 1, "k1_S2": 2, "k1_S4": 4, "k0_S1": 1}.get(r.get("struct"), None)
    rows.append(dict(method=m, target=r["target"], role="control" if r["target"] in CONTROLS else ("common" if r["target"] in COMMON else "diagnostic"),
                     struct=r.get("struct", r.get("structure")), S=S, complexity=cx, **units, file_bytes=r["export"]["file_bytes"],
                     best_baseline_bytes=r["baselines_bytes"]["best"], best_baseline=r["baselines_bytes"]["best_method"],
                     exact=e["exact"], mean_err=round(e["mean_err"], 4), hold=round(e["mean_hold"], 4), ratio_to_hold=round(e["mean_ratio_to_hold"], 3),
                     change_iou=round(e["change_iou"], 3), tol_space=round(e["mean_tol_space"], 4), tol_time=round(e["mean_tol_time"], 4), horizon=e["horizon"], T=e["T"],
                     final_err=round(e["final_err"], 4), final_hold=round(e["final_hold"], 4),
                     search_cost=(f"{r['search_cpu_s']} CPU-s" if "search_cpu_s" in r else f"{r['train_s']} GPU-s train + {r['compile_s']} s compile"),
                     node_replay_equals_P=r["export"]["node_replay_equals_P"], python_replay_equals_P=r["export"]["python_replay_equals_P"],
                     node_decode_ms=r["export"]["node_decode_ms"], media=r.get("media"), result=os.path.relpath(p, ROOT)))
json.dump(rows, open(os.path.join(C7, "results.json"), "w"), indent=1)
md = []
for role in ("control", "common", "diagnostic"):
    md += [f"\n### {role}\n", "| method | target | S | complexity | file B (best baseline B) | exact | mean err | hold | err/hold | change-IoU | tol-space | tol-time | horizon/T | search cost | Node replay |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted((x for x in rows if x["role"] == role), key=lambda x: (x["target"], x["method"], x["mean_err"])):
        md.append(f"| {r['method']} | {r['target']} | {r['S']} | {r['complexity']} | {r['file_bytes']} ({r['best_baseline_bytes']} {r['best_baseline']}) | {r['exact']} | {r['mean_err']} | {r['hold']} | "
                  f"{r['ratio_to_hold']} | {r['change_iou']} | {r['tol_space']} | {r['tol_time']} | {r['horizon']}/{r['T']} | {r['search_cost']} | {r['node_replay_equals_P']} |")
led = [json.loads(l) for l in open(os.path.join(C7, "cpu_ledger.jsonl"))]
c = collections.Counter(); w = collections.Counter()
for r in led: c[r["kind"]] += r["cpu_s"]; w[r["kind"]] += r["wall_s"]
md.append("\n### CPU ledger (core-s; wall-s separately)\n")
md += [f"- {k}: CPU {c[k]:.0f}, wall {w[k]:.0f}" for k in sorted(c)] + [f"- **total CPU {sum(c.values()) / 3600:.2f} core-h**"]
open(os.path.join(C7, "summary07.md"), "w").write("\n".join(md) + "\n"); print("\n".join(md))
