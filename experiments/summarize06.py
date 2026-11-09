"""Campaign 06 tables (analysis only): runs/c6/{cpu_ledger.jsonl, */*.result.json} ->
runs/c6/results.json (machine-readable) and runs/c6/summary06.md."""
import collections, glob, json, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
C6 = os.path.join(ROOT, "runs", "c6")
led = [json.loads(l) for l in open(os.path.join(C6, "cpu_ledger.jsonl"))]
solve = {r["key"]: r for r in led if r["kind"] == "solve"}
res = {}
for p in glob.glob(os.path.join(C6, "*", "*.result.json")):
    r = json.load(open(p)); res[os.path.relpath(p, ROOT)] = r
rows = []
for key, s in solve.items():
    stem = [p for p, r in res.items() if r["inst"] == s["inst"] and r["solver"] == s["solver"] and (r.get("cube") or None) == s.get("cube")]
    r = res[stem[0]] if stem else {}
    rows.append(dict(phase=os.path.basename(os.path.dirname(stem[0])) if stem else None, inst=s["inst"], target=s["target"], G=s["G"], variant=s["variant"],
                     solver=s["solver"], cube=s.get("cube"), status=r.get("status", s["status"]), solve_cpu_s=r.get("solve_cpu_s"),
                     process_cpu_s=s["cpu_s"], wall_s=s["wall_s"], maxrss_gb=s["maxrss_gb"], cpu_cap_s=s["cpu_cap_s"], cnf_sha256=s["cnf_sha256"],
                     target_sha256=r.get("target_sha256"), proof=r.get("proof"), seed=r.get("seed"),
                     python_rollout_exact=r.get("python_rollout_exact"), node_file_decode_exact=r.get("node_file_decode_exact"),
                     file_bytes=r.get("file_bytes"), rule_bits=r.get("rule_bits"), node_decode_ms=r.get("node_decode_ms"), result_file=stem[0] if stem else None))
cubes = [r for r in rows if r["cube"]]
cube_summary = dict(instance=cubes[0]["inst"] if cubes else None, n_cubes_planned=22, n_done=len(cubes),
                    sat=[r["cube"] for r in cubes if r["status"] == "SAT"], unsat=sorted(r["cube"] for r in cubes if r["status"] == "UNSAT"),
                    unknown=sorted(r["cube"] for r in cubes if r["status"].startswith("UNKNOWN")),
                    covers_whole_space_if_all_unsat=True) if cubes else None
cost = collections.Counter()
for r in led: cost[r["kind"]] += r["cpu_s"]
success = [r for r in rows if r["status"] == "SAT" and r["target"] == "ba32_3000_5" and r["python_rollout_exact"] and r["node_file_decode_exact"]
           and r["file_bytes"] is not None and r["file_bytes"] < 80 and "witness" not in r["variant"]]
out = dict(rows=sorted(rows, key=lambda r: (r["phase"] or "", r["inst"], r["solver"], r["cube"] or "")), cube_summary=cube_summary,
           cpu_core_s_by_kind=dict(cost), cpu_core_h_total=round(sum(cost.values()) / 3600, 3), gpu_h=0,
           compression_success_video_only=success, best_baseline_bytes=80)
json.dump(out, open(os.path.join(C6, "results.json"), "w"), indent=1)
md = ["| phase | instance | solver | cube | status | solve CPU s | process CPU s | peak GB | file bytes | Python | Node |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in out["rows"]:
    md.append(f"| {r['phase']} | {r['inst']} | {r['solver']} | {r['cube'] or ''} | {r['status']} | {r['solve_cpu_s'] or ''} | {r['process_cpu_s']} | {r['maxrss_gb']} | "
              f"{r['file_bytes'] or ''} | {r['python_rollout_exact'] if r['python_rollout_exact'] is not None else ''} | {r['node_file_decode_exact'] if r['node_file_decode_exact'] is not None else ''} |")
md.append(f"\ncube summary: {json.dumps(cube_summary)}\n\nCPU by kind (core-s): {json.dumps(dict(cost))}; total {out['cpu_core_h_total']} core-h; GPU 0")
open(os.path.join(C6, "summary06.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
