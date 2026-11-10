"""Campaign 11 results table (every seed) from runs/c11/eval/*.json -> runs/c11/summary11.md."""
import json, os, glob, statistics as st
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
E = lambda f: json.load(open(os.path.join(ROOT, "runs", "c11", "eval", f)))
groups = [("main", "C10 B+C (10,320) ref", "ref_main.json", "c10_combo10"), ("main", "C9 1,304 distilled (A, historical)", "ref_main.json", "small8x16"),
          ("main", "C11 1,304 direct (B)", "p1_main_8x16.json", "_D_"), ("main", "C11 1,304 distilled (C)", "p1_main_8x16.json", "_T_"),
          ("main", "C11 3,884 direct", "p1_main_12x32.json", ""), ("confirm", "C10 B+C (10,320) ref", "ref_confirm.json", "c10_combo10"),
          ("confirm", "C9 1,304 distilled (A, historical)", "ref_confirm.json", "small8x16"), ("confirm", "C11 1,304 direct (B)", "p1_confirm_8x16.json", ""),
          ("confirm", "C11 3,884 direct", "p1_confirm_12x32.json", ""), ("L120", "C11 10,320 direct", "L120_big.json", ""), ("L120", "C11 3,884 direct", "L120_small.json", "")]
if os.path.exists(os.path.join(ROOT, "runs", "c11", "eval", "L240_big.json")): groups.append(("L240", "C11 10,320 direct (1 seed, exploratory)", "L240_big.json", ""))
L = ["# Campaign 11 results (every seed; full-interval autonomous replay from the export, original time alignment)", "",
     "err = binary pixel error; early/mid/late = error per third; lag1 = tolerant change F1; mass = predicted/true changed pixels (lag 1); "
     "thinFG / thinBG = recall of the original's thin parts / thin gaps; specks = small (<=3 px) fg components pred/true; blips = spurious 1-frame flips; "
     "node = numpy vs independent Node binary mismatching pixels; sust = sustained-motion gate (long intervals).", "",
     "| task | model | export | err | early / mid / late | lag1 | mass | thinFG | thinBG | specks | blips | C9 rule / sust | node px | bytes |", "|" + "---|" * 14]
med = []
for task, lab, f, key in groups:
    rows = [(k, r) for k, r in E(f).items() if key in os.path.basename(k)]
    for k, r in rows:
        a = r["all"]; c = r["small_cc"]
        L.append(f"| {task} | {lab} | {os.path.basename(k)} | {a['err']:.4f} | {' / '.join(f'{x['err']:.4f}' for x in r['thirds'])} | {r['lag1']['f1_tol']:.3f} | {r['lag1']['mass_ratio']:.2f} | "
                 f"{r['thin']['thin_fg_recall']:.3f} | {r['thin']['thin_bg_recall']:.3f} | {c['pred_specks']}/{c['true_specks']} | {r['blips_spurious']} | "
                 f"{r['c9_rule']['pass_C'] if 'c9_rule' in r else r.get('sustained')} | {r['verify'].get('binary_mismatch_px', '-')} | {r['file_bytes']} |")
    if rows: med.append(f"| {task} | {lab} | {len(rows)} | {st.median(r['all']['err'] for _, r in rows):.4f} | {st.median(r['thirds'][2]['err'] for _, r in rows):.4f} | {st.median(r['lag1']['f1_tol'] for _, r in rows):.3f} | {st.median(r['lag1']['mass_ratio'] for _, r in rows):.2f} | {st.median(r['thin']['thin_fg_recall'] for _, r in rows):.3f} | {st.median(r['thin']['thin_bg_recall'] for _, r in rows):.3f} |")
L += ["", "## Medians", "", "| task | model | seeds | err | late | lag1 | mass | thinFG | thinBG |", "|" + "---|" * 9] + med
open(os.path.join(ROOT, "runs", "c11", "summary11.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L[-len(med) - 2:]))
