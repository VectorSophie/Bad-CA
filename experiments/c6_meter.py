"""Campaign 06 CPU meter: run a command, append user+sys CPU of it and all its waited children (os.wait4,
which includes solver threads) plus wall and peak RSS to runs/c6/cpu_ledger.jsonl.
usage: c6_meter.py KIND -- cmd args..."""
import json, os, subprocess, sys, time
LEDGER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "c6", "cpu_ledger.jsonl")


def log(rec):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    kind = sys.argv[1]; cmd = sys.argv[sys.argv.index("--") + 1:]
    w0 = time.time(); p = subprocess.Popen(cmd); _, st, ru = os.wait4(p.pid, 0)
    log(dict(kind=kind, cmd=" ".join(cmd)[:300], cpu_s=round(ru.ru_utime + ru.ru_stime, 1), wall_s=round(time.time() - w0, 1),
             maxrss_gb=round(ru.ru_maxrss / 2 ** 20, 2), status=st, t_end=time.time()))
    sys.exit(os.waitstatus_to_exitcode(st))
