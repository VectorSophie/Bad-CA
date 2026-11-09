"""Campaign 05 tables (analysis only). Reads runs/c5/{cpu_ledger.jsonl, results.json, certs.json, prefix.json};
writes runs/c5/search.json and runs/c5/summary05.md."""
import collections, json, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c5_targets import target
shape = lambda c: "×".join(map(str, target(c).shape))
C5 = os.path.join(ROOT, "runs", "c5")
led = [json.loads(l) for l in open(os.path.join(C5, "cpu_ledger.jsonl"))]
res = json.load(open(os.path.join(C5, "results.json")))
md = []; P = md.append

# ---- search table ----
S = collections.defaultdict(list)
for r in led:
    if r["kind"] == "search":
        S[(r["target"], r["struct"], "v3" if r.get("entries") else ("lex" if r["lex"] else "v1"))].append(r)
search = []
P("### Search (direct circuit synthesis; status per G, CPU s in brackets)\n")
P("| clip | structure | encoding | per G | largest G proved UNSAT | first SAT G | CPU core-s |")
P("|---|---|---|---|---|---|---|")
for (t, st, enc), rs in sorted(S.items()):
    rs = sorted(rs, key=lambda r: r["G"]); stat = {}
    for r in rs: stat.setdefault(r["G"], []).append(r["status"])
    unsat = [g for g, v in stat.items() if "UNSAT" in v]; sat = [g for g, v in stat.items() if any(x.startswith("SAT") for x in v)]
    proved = max(unsat, default=None)  # SAT is monotone in G (buffer padding), so UNSAT at G covers every smaller G
    per = ", ".join(f"G{r['G']} {r['status'].replace('UNKNOWN_cpu_limit', 'UNKNOWN')} ({r['cpu_s']:.0f})" for r in rs)
    cpu = sum(r["cpu_s"] for r in rs)
    search.append(dict(clip=t, struct=st, encoding=enc, runs=rs, max_G_unsat_with_all_smaller=proved, first_sat_G=min(sat) if sat else None, cpu_s=cpu))
    P(f"| {t} | {st} | {enc} | {per} | {proved if proved is not None else '-'} | {min(sat) if sat else '-'} | {cpu:.0f} |")
LB = collections.defaultdict(lambda: dict(unsat=-1, sat=None, unknown=[]))
for x in search:
    b = LB[(x["clip"], x["struct"])]
    if x["max_G_unsat_with_all_smaller"] is not None: b["unsat"] = max(b["unsat"], x["max_G_unsat_with_all_smaller"])
    if x["first_sat_G"] is not None: b["sat"] = x["first_sat_G"] if b["sat"] is None else min(b["sat"], x["first_sat_G"])
    b["unknown"] += [r["G"] for r in x["runs"] if r["status"].startswith("UNKNOWN")]
P("\n### Gate-count bounds per clip x structure (all encodings combined; v1/lex/v3 have the same solution existence)\n")
P("| clip | structure | proved: no circuit with <= G gates | smallest SAT G found | minimal? | largest UNKNOWN G probed |")
P("|---|---|---|---|---|---|")
bounds = []
for (t, st), b in sorted(LB.items()):
    minimal = b["sat"] is not None and b["unsat"] == b["sat"] - 1
    bounds.append(dict(clip=t, struct=st, no_circuit_up_to_G=b["unsat"] if b["unsat"] >= 0 else None, sat_G=b["sat"], minimal_proved=minimal,
                       max_unknown_G=max(b["unknown"], default=None)))
    P(f"| {t} | {st} | {b['unsat'] if b['unsat'] >= 0 else '-'} | {b['sat'] if b['sat'] is not None else '-'} | {'yes' if minimal else 'no'} | {max(b['unknown'], default='-')} |")
json.dump(dict(per_series=search, bounds=bounds), open(os.path.join(C5, "search.json"), "w"), indent=1)

# ---- files ----
rows = res["rows"]
P("\n### Total file bytes per clip (every file decoded from its bytes; CA files also by the Node player)\n")
P("| clip | T×H×W | changelist | xor_rle | frame_rle | zlib | bz2 | lzma | best LUT-route file (bytes, flow) | best LUT-route circuit (bytes, gates) | best direct-synthesis circuit (bytes, G) | CA beats every tested baseline? |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|")
clips = []
for r in rows:
    if r["clip"] not in clips: clips.append(r["clip"])
verdict = {}
for c in clips:
    R = [r for r in rows if r["clip"] == c]
    base = {r["method"]: r["bytes"] for r in R if r["route"] == "baseline"}
    lut = min((r for r in R if r["route"] == "lut_then_reduce"), key=lambda r: r["bytes"], default=None)
    lutc = min((r for r in R if r["route"] == "lut_then_reduce" and r["method"] == "ca_circuit"), key=lambda r: r["bytes"], default=None)
    d = min((r for r in R if r["route"] == "direct_synthesis"), key=lambda r: r["bytes"], default=None)
    best_ca = min([x["bytes"] for x in (lut, d) if x] or [None]) if (lut or d) else None
    win = best_ca is not None and best_ca < min(base.values())
    verdict[c] = dict(best_baseline=min(base, key=base.get), best_baseline_bytes=min(base.values()), best_lut_route_bytes=lut and lut["bytes"],
                      best_direct_bytes=d and d["bytes"], ca_beats_all_tested_baselines=win)
    P(f"| {c} | {shape(c)} | " + " | ".join(str(base[m]) for m in ("changelist", "xor_rle", "frame_rle", "zlib", "bz2", "lzma")) +
      f" | {f'{lut['bytes']} ({lut['flow']}, {lut['level']})' if lut else '-'} | {f'{lutc['bytes']} ({lutc['gates']})' if lutc else '-'} | "
      f"{f'{d['bytes']} (G={d['G']}, {d['struct']})' if d else '-'} | {'**yes**' if win else 'no'} |")
json.dump(verdict, open(os.path.join(C5, "verdicts.json"), "w"), indent=1)

# ---- decode times ----
P("\n### Decode cost (CA files): updates and time\n")
P("| clip | file | bytes | updates (cells × generations) | Python decode s | Node decode ms |")
P("|---|---|---|---|---|---|")
for c in clips:
    for r in sorted((r for r in rows if r["clip"] == c and r.get("method", "").startswith("ca_")), key=lambda r: r["bytes"])[:2]:
        P(f"| {c} | {r['name']} | {r['bytes']} | {r['updates']} | {r['python_decode_s']} | {r['node_decode_ms']} |")

# ---- ledger ----
tot = collections.Counter()
for r in led: tot[r["kind"] if r["kind"] != "search" else "search:" + r.get("outdir", "")] += r["cpu_s"]
P("\n### CPU ledger (core-s, user+sys incl. children)\n")
for k, v in sorted(tot.items()): P(f"- {k}: {v:.0f}")
P(f"- **total: {sum(tot.values()):.0f} core-s = {sum(tot.values()) / 3600:.2f} core-h**; GPU: 0")
open(os.path.join(C5, "summary05.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
