"""Builds the CHECKPOINT_03 tables from runs/exp06 and runs/exp07 (analysis only, no search)."""
import json, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = lambda p: json.load(open(p))

print("### A. Length scaling (exact only; ba32_4500, F_priv k=1 S=2, 30 min CPU each)\n")
print("| frames | changed px | status | encode s | first sol. (s, E) | best E (s) | rule bits (keep+exc) | changelist | xor_rle | frame_rle | rule / best recording | Node exact |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for L in (5, 10, 20, 40):
    p = os.path.join(ROOT, "runs", "exp06", f"ba32_4500_{L}_E.result.json")
    if not os.path.exists(p): print(f"| {L} | missing |"); continue
    r = R(p); tr = r.get("trace", [])
    rec = r.get("recording_bits", {}); best = min(rec.values()) if rec else None
    rb = r.get("rule_bits")
    print(f"| {L} | {r.get('changed_pixels')} | {r['status']} | {r.get('encode_s')} | "
          f"{(tr[0]['t'], tr[0]['exceptions']) if tr else '-'} | {(tr[-1]['exceptions'], tr[-1]['t']) if tr else '-'} | {rb or '-'} | "
          f"{rec.get('changelist')} | {rec.get('xor_rle')} | {rec.get('frame_rle')} | {round(rb / best, 2) if rb and best else '-'} | "
          f"{r.get('node_hash_matches_target', '-')} |")

print("\n### B. Hard-rule optimisation, ba32_3000_5 E (selection by hard autonomous rollout)\n")
print("| id | track | exact (seeds) | best hard err | change-IoU | live gates | circuit bits | LUT-exc bits | own-output best recording bits | steps | budget |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
rows = []
for m in ("R0", "HG", "HS", "HH", "HB", "T1", "T2"):
    p = os.path.join(ROOT, "runs", "exp07", f"{m}_main", "result.json")
    if not os.path.exists(p): print(f"| {m} | missing |"); continue
    r = R(p); s = r["selected"]; ne = sum(x["exact"] for x in r["per_seed"])
    print(f"| {m} | {r['track']} | {ne}/{len(r['per_seed'])} | {s['hard_pixel_err']:.4f} | {s['change_iou']:.2f} | {s['live_gates']} | "
          f"{s['circuit_bits']} | {s['lut_exception_bits']} | {min(s['own_output_recording_bits'].values())} | {r['compute']['steps']} | "
          f"{r['opt']['budget_s']:.0f} s GPU |")
    rows.append((m, r))
p = os.path.join(ROOT, "runs", "exp07", "D1_main", "result.json")
if os.path.exists(p):
    r = R(p)
    print(f"| D1 | video only | {'1/1' if r['exact'] else '0/1'} | {r['hard_pixel_err']:.4f} | {r['change_iou']:.2f} | {r['live_gates']} | "
          f"{r['circuit_bits']} | {r['lut_exception_bits']} | {min(r['own_output_recording_bits'].values())} | {r['evals']} evals | {r['budget_s']:.0f} s CPU |")
p = os.path.join(ROOT, "runs", "exp07", "D2_main", "ba32_3000_5_E.result.json")
if os.path.exists(p):
    r = R(p)
    print(f"| D2 | video only (SAT repair) | {r['status']} | {'0' if r.get('python_rollout_exact') else '-'} | - | - | - | {r.get('rule_bits')} | - | "
          f"hamming to R0 LUT: {r.get('best_exceptions')} | 600 s CPU |")
print("\nhold-frame baseline err:", rows[0][1]["hold_pixel_err"] if rows else "-",
      "; target recording bits:", rows[0][1]["target_recording_bits"] if rows else "-")

print("\n### B2. Where the soft/hard gap comes from (selected checkpoint; errors over frames 1..T-1)\n")
print("| id | soft g + soft s | soft g + round s | soft g + round hidden | hard g + soft s | hard g + round s (=hard) | one-step disagreement (vis, hid) | per-frame soft | per-frame hard |")
print("|---|---|---|---|---|---|---|---|---|")
for m, r in rows:
    g = r["selected"]["gaps"]
    f = lambda x: "[" + ", ".join(f"{v:.3f}" for v in x) + "]"
    print(f"| {m} | {g['soft_gates_soft_state']:.4f} | {g['soft_gates_round_state']:.4f} | {g['soft_gates_round_hidden']:.4f} | "
          f"{g['hard_gates_soft_state']:.4f} | {g['hard_gates_round_state']:.4f} | "
          f"{g['one_step_disagreement_per_channel'][0]:.4f}, {g['one_step_disagreement_per_channel'][1]:.4f} | {f(g['per_frame_soft'])} | {f(g['per_frame_hard'])} |")
