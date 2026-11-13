"""GPU ledger for this campaign from the shared queue: one row per gpuq job (label, GPUs, start, end, wall, exit),
GPU-hours = assigned GPUs x wall time. usage: gen_ledger.py ID [ID ...]  -> runs/gen/gpu_ledger.jsonl (rewritten)"""
import json, os, re, subprocess, sys
from datetime import datetime
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "gen", "gpu_ledger.jsonl")
rows = []
for jid in sys.argv[1:]:
    info = subprocess.run(["gpuq", "-i", jid], capture_output=True, text=True).stdout
    g = lambda k: (re.search(rf"^{k}:\s*(.*)$", info, re.M) or [None, None])[1]
    st, en = g("Start time"), g("End time"); fmt = "%a %b %d %H:%M:%S %Y"
    wall = (datetime.strptime(en, fmt) - datetime.strptime(st, fmt)).total_seconds() if st and en else None
    lab = re.search(r"\[(gen-[^\]]+)\]", subprocess.run(["gpuq"], capture_output=True, text=True).stdout.split(f"\n{jid} ")[1] if f"\n{jid} " in subprocess.run(["gpuq"], capture_output=True, text=True).stdout else "")
    rows.append(dict(job=int(jid), label=lab.group(1) if lab else None, start=st, end=en, wall_s=wall, gpus=1, gpu_h=round(wall / 3600, 4) if wall else None,
                     exit=g("Exit status"), cmd=info.strip().splitlines()[-1][:200]))
with open(OUT, "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print("total GPU-h", round(sum(r["gpu_h"] or 0 for r in rows), 3)); [print(r["job"], r["label"], r["gpu_h"], r["exit"]) for r in rows]
