"""Exp15 (Campaign 05): every method as a decodable file, sizes in bytes, independent decoding.

Per clip:
  baselines   change list, XOR+RLE, frame RLE, zlib, bz2, lzma (ca.container), Python-decoded;
  LUT route   every exact rule of CHECKPOINT_02-04 for that clip (keep-exception LUT, full and reach
              level) and every ABC circuit from exp11/exp11_c4 (full and reach), as container files;
  direct      every SAT circuit of Campaign 05 (runs/c5/**/ *.result.json with status SAT).
Every CA file is decoded from its bytes in Python AND by player/decode_file.mjs (Node); the SHA-256
of all T frames must equal the target's. Full truth-table equivalence is recorded only where a
reference function exists (LUT-route conversions, from exp11). Writes runs/c5/results.json and
runs/c5/files/<clip>/*.
usage: exp15_files.py
"""
import base64, glob, json, os, subprocess, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import container, costs
from ca.rule import family_taps
from c5_targets import target

RUNS = os.path.join(ROOT, "runs"); OUT = os.path.join(RUNS, "c5", "files")
CLIPS = ["pcA", "pcB", "translate24", "disc12", "bar12", "ba32_3000_5", "ba32_4500_5", "ba32_4500_10"]


def fam_of(taps):
    n = len(taps); k = 0 if n == 9 else (n - 9 if all(t[1:] == [0, 0] for t in taps[9:]) else (n - 9) // 9)
    fam = "memoryless" if k == 0 else ("F_priv" if n == 9 + k else "F_shared")
    assert [list(t) for t in family_taps(k, "moore" if fam == "F_shared" else "self")] == [list(t) for t in taps]
    return fam, k


def node(path):
    t0 = time.perf_counter()
    r = json.loads(subprocess.run(["node", os.path.join(ROOT, "player", "decode_file.mjs"), path], capture_output=True, text=True, check=True).stdout)
    r["node_wall_s"] = round(time.perf_counter() - t0, 3); return r


def ca_file(clip, F, name, method, rule, extra):
    os.makedirs(os.path.join(OUT, clip), exist_ok=True)
    data = container.encode(method, F, rule); p = os.path.join(OUT, clip, name + ".cab"); open(p, "wb").write(data)
    t0 = time.perf_counter(); m, G, info = container.decode(open(p, "rb").read()); py_s = time.perf_counter() - t0
    nd = node(p)
    return dict(clip=clip, name=name, method=method, bytes=len(data), python_decode_exact=bool((G == F).all()), python_decode_s=round(py_s, 3),
                node_decode_exact=nd["sha256"] == container.sha(F), node_decode_ms=nd["decode_ms"], updates=info["updates"],
                S=info["S"], k=info["k"], family=info["family"], gates=info.get("gates"), file=os.path.relpath(p, ROOT), **extra)


def norm(p):
    return os.path.normpath(os.path.join(RUNS, p[6:])) if p.startswith("../../") else os.path.join(ROOT, p)


rows = []
for clip in CLIPS:
    F = target(clip)
    for m, data in container.baselines(F).items():
        os.makedirs(os.path.join(OUT, clip), exist_ok=True); p = os.path.join(OUT, clip, m + ".bin"); open(p, "wb").write(data)
        t0 = time.perf_counter(); _, G, _ = container.decode(open(p, "rb").read())
        rows.append(dict(clip=clip, route="baseline", name=m, method=m, bytes=len(data), python_decode_exact=bool((G == F).all()),
                         python_decode_s=round(time.perf_counter() - t0, 3), file=os.path.relpath(p, ROOT),
                         payload_bits_aux=len(dict(changelist=costs.changelist, xor_rle=costs.xor_rle, frame_rle=costs.frame_rle)[m](F)) if m in ("changelist", "xor_rle", "frame_rle") else None))

# LUT -> reduction route (CHECKPOINT_02-04 exact rules, exp11 conversions)
for r in json.load(open(os.path.join(RUNS, "exp11_all", "results.json"))):
    if not r["exact"] or r["target"] not in CLIPS: continue
    F = target(r["target"]); label = r["rule"].replace("../../", "").replace("runs/", "").replace("/", "__").replace(".rule.json", "")
    for lv in ("full", "reach"):
        L = r[lv]
        for kind, ent in [("lut", L["lut_keep_exceptions"])] + [(f, c) for f, c in L["circuits"].items()]:
            ex = json.load(open(norm(ent["export"]))); taps = [list(t) for t in ex["taps"]]; fam, k = fam_of(taps)
            base = dict(route="lut_then_reduce", level=lv, source_rule=r["rule"], flow=kind,
                        full_tt_equal=ent.get("full_tt_equal") if lv == "full" else "n/a (reach: don't-cares)")
            if kind == "lut":
                n = len(taps)
                luts = np.stack([np.unpackbits(np.frombuffer(base64.b64decode(b), np.uint8), bitorder="little")[:2 ** n] for b in ex["luts_b64"]])
                rows.append(ca_file(r["target"], F, f"{label}.{lv}.lut", "ca_lut_exc", dict(luts=luts, taps=[tuple(t) for t in taps], family=fam, k=k, S=ex["steps_per_frame"]), base))
            else:
                rows.append(ca_file(r["target"], F, f"{label}.{lv}.{kind}", "ca_circuit",
                                    dict(gates=[tuple(g) for g in ex["gates"]], outputs=ex["outputs"], n_in=len(taps), family=fam, k=k, S=ex["steps_per_frame"]), base))

# direct synthesis (Campaign 05)
impl_errors = []
for p in sorted(glob.glob(os.path.join(RUNS, "c5", "**", "*.result.json"), recursive=True)):
    r = json.load(open(p))
    if r.get("status") != "SAT" or r["target"] not in CLIPS: continue
    if not r.get("python_rollout_exact"):  # pilot circuits decoded with the inverted truth-table bug (see CAMPAIGN_05)
        impl_errors.append(os.path.relpath(p, ROOT)); continue
    F = target(r["target"])
    rows.append(ca_file(r["target"], F, os.path.basename(p).replace(".result.json", ""), "ca_circuit",
                        dict(gates=[tuple(g) for g in r["gates"]], outputs=r["outputs"], n_in=r["n_inputs"], family=r["family"], k=r["k"], S=r["S"]),
                        dict(route="direct_synthesis", struct=r["struct"], G=r["G"], symmetry=r["symmetry"], search_dir=os.path.relpath(os.path.dirname(p), ROOT),
                             full_tt_equal="n/a (no reference function)")))

bad = [x for x in rows if not x["python_decode_exact"] or ("node_decode_exact" in x and not x["node_decode_exact"])]
json.dump(dict(rows=rows, n_rows=len(rows), all_decodes_exact=not bad, failures=bad, excluded_implementation_errors=impl_errors), open(os.path.join(RUNS, "c5", "results.json"), "w"), indent=1)
print(len(rows), "rows; decode failures:", len(bad))
for clip in CLIPS:
    best = lambda route: min((x for x in rows if x["clip"] == clip and x["route"] == route), key=lambda x: x["bytes"], default=None)
    b, l, d = best("baseline"), best("lut_then_reduce"), best("direct_synthesis")
    print(clip, "baseline", b and (b["name"], b["bytes"]), "| lut route", l and (l["name"], l["bytes"]), "| direct", d and (d["name"], d["bytes"]))
