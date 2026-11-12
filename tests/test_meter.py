"""Integration check for the CPU meter and the SAT time limit (round-4 pre-flight; FINDINGS §9a defect).
1. solve() returns TIMEOUT within its wall limit on a hard instance.
2. A wall-killed Pool run leaves no worker alive, books the workers' CPU, and keeps rows written before the kill.
3. SIGTERM to the meter does the same."""
import json, os, signal, subprocess, sys, tempfile, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); PY = sys.executable
sys.path.insert(0, os.path.join(ROOT, "experiments")); sys.path.insert(0, ROOT)
WORK = r'''
import sys, time, json
from multiprocessing import Pool
def burn(i):
    t = time.process_time()
    while time.process_time() - t < 60: pass
if __name__ == "__main__":
    open(sys.argv[1], "a").write(json.dumps({"row": 0}) + "\n")   # a partial result written before the kill
    with Pool(3) as p: p.map(burn, range(3))
'''


def alive(tag):
    out = subprocess.run(["pgrep", "-f", tag], capture_output=True, text=True).stdout.split()
    return [q for q in out if int(q) != os.getpid()]


def metered(tmp, tag, wall, term_after=None):
    led, rows, script = (os.path.join(tmp, n) for n in ("ledger.jsonl", "rows.jsonl", tag + ".py"))
    open(script, "w").write(WORK)
    m = subprocess.Popen([PY, os.path.join(ROOT, "experiments", "c9_meter.py"), "t", "--wall", str(wall), "--", PY, script, rows],
                         env=dict(os.environ, CA_LEDGER=led))
    if term_after: time.sleep(term_after); m.send_signal(signal.SIGTERM)
    m.wait(timeout=60); time.sleep(1)
    rec = [json.loads(l) for l in open(led)][-1]
    assert not alive(script), f"workers left after {tag}: {alive(script)}"
    assert rec["wall_killed"] and rec["procs"] >= 4, rec
    assert rec["cpu_s"] >= 3 * (rec["wall_s"] - 2.5), rec   # three busy workers were counted
    assert open(rows).read().strip(), "partial rows lost"
    return rec


if __name__ == "__main__":
    import numpy as np, rr_scale
    from exp45_c12 import target
    F = target("0:6572").astype(np.uint8)[4500:4512]
    t = time.time(); r = rr_scale.solve(F, "full", 3, 1, lambda y, x: (y, x), timeout=3); dt = time.time() - t
    assert r["status"] == "TIMEOUT" and r["solve_wall_s"] < 6, (r["status"], r["solve_wall_s"]); print("sat time limit ok", round(dt, 1), "s")
    with tempfile.TemporaryDirectory() as tmp:
        print("wall kill ok", metered(tmp, "meterwall", 5))
        print("sigterm ok", metered(tmp, "metersig", 600, term_after=5))
    print("ALL OK")
