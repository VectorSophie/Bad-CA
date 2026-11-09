"""Builds the CHECKPOINT_04 tables (analysis only, no search).
usage: summarize04.py EXP09_DIR EXP10_DIR EXP11_DIR EXP12_DIR"""
import glob, json, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RUNS = os.path.join(ROOT, "runs")
d09, d10, d11, d12 = sys.argv[1:5]
J = lambda p: json.load(open(p))

print("### Part 1: incremental vs cold-start SAT (ba32_4500, F_priv k=1 S=2, first exact rule)\n")
arms = {}
for p in sorted(glob.glob(os.path.join(RUNS, d09, "*.jsonl"))):
    recs = [json.loads(l) for l in open(p)]
    arm = recs[0]["arm"]
    arms[arm] = {r["L"]: r for r in recs if r.get("event") == "solved"}
cold = {L: arms.get(f"cold_keep_L{L}", {}).get(L) for L in (5, 10, 20, 40)}
inc = sorted(a for a in arms if a.startswith("incr"))
hdr = "| L | cold step s | cold sum over 5..L s | " + " | ".join(f"{a} step s / cum s" for a in inc) + " |"
print(hdr); print("|" + "---|" * (hdr.count("|") - 1))
csum, cok = 0.0, True
for L in (5, 10, 20, 40):
    c = cold[L]
    if c and cok: csum += c["step_s"]
    else: cok = False
    row = [str(L), f"{c['step_s']} ({c['status']})" if c else "UNKNOWN_timeout", f"{csum:.1f}" if cok else "UNKNOWN"]
    for a in inc:
        r = arms[a].get(L)
        row.append(f"{r['step_s']} / {r['cum_s']} ({r['status']})" if r else "UNKNOWN_timeout")
    print("| " + " | ".join(row) + " |")
print("\nlongest length reached per incremental arm:", {a: max(arms[a]) if arms[a] else None for a in inc})
print("peak RSS GB:", {a: max((r["maxrss_gb"] for r in arms[a].values()), default=None) for a in arms})

print("\n### Part 2: SAT repair from different starts (warm start = phase preference + Hamming objective, NOT a hard constraint)\n")
print("| clip | start | status | first exact s (encode s) | repair distance at first exact | best repair-distance UPPER BOUND (s) | keep-exc rule bits | Node exact | GPU cost of start |")
print("|---|---|---|---|---|---|---|---|---|")
for p in sorted(glob.glob(os.path.join(RUNS, d10, "*", "*.result.json"))):
    if "__" not in os.path.basename(os.path.dirname(p)): continue
    r = J(p); start = os.path.basename(os.path.dirname(p)).split("__")[1]; tr = r.get("trace", [])
    sdir = os.path.join(RUNS, d10, "starts_" + r["target"])
    gpu = "-"
    src = os.path.join(sdir, f"{start}.source_result.json")
    if os.path.exists(src):
        s = J(src); gpu = f"{s['compute']['train_wall_s'] + s['compute']['compile_s']:.0f} GPU-s train+compile ({len(s['seeds'])} seeds vmapped); selected ckpt at {s['selected']['t']} s"
    print(f"| {r['target']} | {start} | {r['status']} | {tr[0]['t'] if tr else '-'} ({r.get('encode_s')}) | {tr[0]['exceptions'] if tr else '-'} | "
          f"{(tr[-1]['exceptions'], tr[-1]['t']) if tr else '-'} | {r.get('rule_bits', '-')} | {r.get('node_hash_matches_target', '-')} | {gpu} |")

print("\n### Part 3: exact rules -> circuits (bits include gate ops, wiring, outputs, constants)\n")
print("| rule | target | consulted | full: LUT-exc bits | full: best circuit bits (gates, flow) | reach: LUT-exc bits | reach: best circuit bits (gates, flow) | best recording | all checks |")
print("|---|---|---|---|---|---|---|---|---|")
for r in J(os.path.join(RUNS, d11, "results.json")):
    if not r["exact"]: continue
    f, a = r["full"], r["reach"]
    ok = all(v["decoded_python_rollout_exact"] and v["node_replay_matches_target"] for lv in (f, a)
             for v in list(lv["circuits"].values()) + [lv["lut_keep_exceptions"]])
    bc = lambda lv: f"{lv['best_circuit']['bits']} ({lv['best_circuit']['gates']}, {lv['best_circuit']['flow']})"
    print(f"| {r['rule'].replace('runs/', '')} | {r['target']} | {r['consulted_entries']} | {f['lut_keep_exceptions']['bits']} | {bc(f)} | "
          f"{a['lut_keep_exceptions']['bits']} | {bc(a)} | {min(r['recording_bits'].values())} | {ok} |")

print("\n### Part 4: binarization penalty (selection by hard autonomous rollout; penalty/soft loss are NOT success criteria)\n")
print("| run | exact seeds | best hard err | hold err | selected it / t | penalty at selection | video sq loss at selection | steps |")
print("|---|---|---|---|---|---|---|---|")
for p in sorted(glob.glob(os.path.join(RUNS, d12, "*_main", "result.json"))) + [os.path.join(RUNS, "exp07", "R0_main", "result.json")]:
    r = J(p); s = r["selected"]; log = J(os.path.join(os.path.dirname(p), "log.json"))
    at = [x for x in log if x["seed"] == s["seed"] and x["it"] == s["it"]]
    pen = at[0].get("penalty", "-") if at else "-"; vl = at[0].get("video_sq_loss", "-") if at else "-"
    print(f"| {os.path.relpath(os.path.dirname(p), RUNS)} ({r['target']}) | {sum(x['exact'] for x in r['per_seed'])}/{len(r['per_seed'])} | "
          f"{s['hard_pixel_err']:.4f} | {r['hold_pixel_err']:.4f} | {s['it']} / {s['t']} | {pen if pen == '-' else round(pen, 1)} | "
          f"{vl if vl == '-' else round(vl, 1)} | {r['compute']['steps']} |")
