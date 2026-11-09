"""Campaign 05 driver: G-ascending series of exp13 workers with CPU accounting.

Each job: prlimit --as=<mem> --cpu=<cap> python exp13_csynth.py ... (prlimit execs, so the waited pid
is the worker). os.wait4 gives user+sys CPU of the worker and its children; appended to
runs/c5/cpu_ledger.jsonl together with wall, maxrss, status. Wall cap = 3 x CPU cap (KILL).
A series stops at the first SAT (or at G_max). Statuses: SAT / UNSAT / UNKNOWN_cpu_limit /
UNKNOWN_wall_limit / UNKNOWN_memout / ERROR.
usage: c5_run.py PLAN.json   (list of {target, struct, G: [..], cpu_s, mem_gb, outdir, lex})
"""
import json, os, signal, subprocess, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LEDGER = os.path.join(ROOT, "runs", "c5", "cpu_ledger.jsonl")
STOP_CORE_H = 9.5
lock = threading.Lock()


def ledger_core_h():
    if not os.path.exists(LEDGER): return 0.0
    return sum(json.loads(l)["cpu_s"] for l in open(LEDGER)) / 3600


def log(rec):
    with lock:
        os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
        open(LEDGER, "a").write(json.dumps(rec) + "\n")


def job(t, st, G, cpu, mem, outdir, lex, entries=False):
    stem = os.path.join(ROOT, "runs", outdir, f"{t.replace(':', '_')}__{st}__G{G}{'_lex' if lex else ''}{'_ent' if entries else ''}")
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    cmd = ["prlimit", f"--as={int(mem * 2 ** 30)}", f"--cpu={int(cpu)}:{int(cpu) + 5}", sys.executable,
           os.path.join(ROOT, "experiments", "exp13_csynth.py"), t, st, str(G), outdir] + (["--lex"] if lex else []) + (["--entries"] if entries else [])
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    w0 = time.time()
    with open(stem + ".log", "w") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        killed = False
        while True:
            pid, rc, ru = os.wait4(p.pid, os.WNOHANG)
            if pid: break
            if time.time() - w0 > 3 * cpu:
                os.kill(p.pid, signal.SIGKILL); killed = True
            time.sleep(0.5)
    p.returncode = rc
    wall = time.time() - w0; cpu_s = ru.ru_utime + ru.ru_stime
    status = None
    if os.path.exists(stem + ".result.json"):
        status = json.load(open(stem + ".result.json"))["status"]
    else:
        txt = open(stem + ".log").read()
        if killed: status = "UNKNOWN_wall_limit"
        elif cpu_s >= cpu - 1: status = "UNKNOWN_cpu_limit"
        elif "MemoryError" in txt or "bad_alloc" in txt or "Cannot allocate" in txt: status = "UNKNOWN_memout"
        else: status = "ERROR"
        if os.path.exists(stem + ".sat.json"): status += "+SAT_saved"
    rec = dict(kind="search", target=t, struct=st, G=G, lex=lex, entries=entries, status=status, cpu_s=round(cpu_s, 1), wall_s=round(wall, 1),
               maxrss_gb=round(ru.ru_maxrss / 2 ** 20, 2), cpu_cap_s=cpu, mem_gb=mem, rc=rc, outdir=outdir, t_end=time.time())
    log(rec); print(json.dumps(rec), flush=True)
    return status


def series(s):
    for G in s["G"]:
        if ledger_core_h() >= STOP_CORE_H:
            print(json.dumps(dict(skipped=s["target"], G=G, reason="ledger stop rule")), flush=True); return
        st = job(s["target"], s["struct"], G, s["cpu_s"], s.get("mem_gb", 16), s["outdir"], s.get("lex", False), s.get("entries", False))
        if st.startswith("SAT") or st.startswith("ERROR"):
            return


if __name__ == "__main__":
    plan = json.load(open(sys.argv[1]))
    with ThreadPoolExecutor(min(16, len(plan))) as ex:
        list(ex.map(series, plan))
