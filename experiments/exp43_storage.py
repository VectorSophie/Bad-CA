"""Exp43 (Campaign 11): storage of final candidates. For each export: write ncab f32 and f16, RELOAD each file, replay
autonomously (float32 state/compute), report the byte breakdown and replay quality; plus the JSON presentation size and a
same-convention recording baseline of the target frames (1 bit/px packed, then zlib-9; lzma-9e for reference).
usage: exp43_storage.py OUT_JSON TARGET EXPORT [EXPORT ...]"""
import json, lzma, os, sys, zlib
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import ncab
from ca.pipeline import load_export
from exp40_c11 import target
from exp41_c11_eval import evaluate

if __name__ == "__main__":
    out, tname, paths = sys.argv[1], sys.argv[2], sys.argv[3:]; F = target(tname)
    packed = np.packbits(F.astype(np.uint8).ravel(), bitorder="little").tobytes()
    res = dict(target=tname, frames=list(F.shape), recording=dict(packed_bits=len(packed), zlib9=len(zlib.compress(packed, 9)),
               lzma9e=len(lzma.compress(packed, preset=9 | lzma.PRESET_EXTREME)), note="lossless recording of the same binary frames; reference only"), models={})
    sd = os.path.join(ROOT, "runs", "c11", "storage"); os.makedirs(sd, exist_ok=True)
    for p in paths:
        hd, P, f0 = load_export(p); name = os.path.basename(p).split(".")[0]; row = dict(json_bytes=os.path.getsize(p) if p.endswith(".json") else None)
        for dt in ("f32", "f16"):
            q = os.path.join(sd, f"{name}_{dt}.ncab"); sizes = ncab.write(q, P, f0, hd["T"], hd["S"], dict(src=os.path.relpath(p, ROOT)), dt)
            r, B, V = evaluate(q, tname, node=True)
            row[dt] = dict(file=os.path.relpath(q, ROOT), sizes=sizes, err=r["all"]["err"], err_band=r["all"]["err_band"], thirds=[x["err"] for x in r["thirds"]],
                           lag1_f1=r["lag1"]["f1_tol"], thin=r["thin"], node=r["verify"], B_sha=__import__("hashlib").sha256(np.ascontiguousarray(B).tobytes()).hexdigest())
        row["f16_binary_identical_to_f32"] = row["f16"]["B_sha"] == row["f32"]["B_sha"]
        res["models"][os.path.relpath(p, ROOT)] = row
        print(name, "f32", row["f32"]["sizes"]["file"], "B err %.4f" % row["f32"]["err"], "| f16", row["f16"]["sizes"]["file"], "B err %.4f" % row["f16"]["err"],
              "identical", row["f16_binary_identical_to_f32"], "| node mism", row["f32"]["node"]["binary_mismatch_px"], row["f16"]["node"]["binary_mismatch_px"], flush=True)
    print("recording", res["recording"]); json.dump(res, open(out, "w"), indent=1, default=float)
