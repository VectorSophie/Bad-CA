"""Campaign 09 CPU meter: run a command, append user+sys CPU of it and all its waited children (os.wait4;
includes threads) plus wall and peak RSS to runs/c9/cpu_ledger.jsonl. Limits: 1 BLAS/OMP thread,
prlimit --as 16 GB unless the command sets its own. usage: c7_meter.py KIND [--wall S] -- cmd args..."""
import json, os, signal, subprocess, sys, time
LEDGER = os.environ.get("CA_LEDGER") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "c9", "cpu_ledger.jsonl")  # Campaign 10 sets CA_LEDGER


TICK = os.sysconf("SC_CLK_TCK")


def session_cpu(sid, seen):
    """Record user+sys CPU of every live process in session sid into seen {pid: s}. Catches pool workers that a kill
    would orphan, whose time wait4 never reports."""
    for d in os.listdir("/proc"):
        if not d.isdigit(): continue
        try: f = open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()
        except OSError: continue
        if int(f[3]) == sid: seen[int(d)] = max(seen.get(int(d), 0), (int(f[11]) + int(f[12])) / TICK)


def log(rec):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    kind = sys.argv[1]; cmd = sys.argv[sys.argv.index("--") + 1:]
    wall = float(sys.argv[sys.argv.index("--wall") + 1]) if "--wall" in sys.argv[:sys.argv.index("--")] else 7200
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    if cmd[0] != "prlimit":
        cmd = ["prlimit", f"--as={16 * 2 ** 30}"] + cmd
    for sg in (signal.SIGTERM, signal.SIGINT):   # a killed meter still kills the group and books its CPU
        signal.signal(sg, lambda *_: globals().__setitem__("wall", -1))
    w0 = time.time(); p = subprocess.Popen(cmd, env=env, start_new_session=True); killed = False; seen = {}
    while True:
        session_cpu(p.pid, seen)
        pid, st, ru = os.wait4(p.pid, os.WNOHANG)
        if pid: break
        if time.time() - w0 > wall and not killed: os.killpg(p.pid, signal.SIGKILL); killed = True   # whole group: pool workers would otherwise run on, unmetered
        time.sleep(0.5)
    if killed:   # leftovers that escaped the group kill (none expected)
        live = {}; session_cpu(p.pid, live); [seen.__setitem__(q, max(seen.get(q, 0), c)) for q, c in live.items()]
        for q in live:   # only pids still in this session: never a reused pid
            try: os.kill(q, signal.SIGKILL)
            except OSError: pass
    p.returncode = st
    log(dict(kind=kind, cmd=" ".join(cmd)[:300], cpu_s=round(max(ru.ru_utime + ru.ru_stime, sum(seen.values())), 1), cpu_waited_s=round(ru.ru_utime + ru.ru_stime, 1),
             cpu_sampled_s=round(sum(seen.values()), 1), procs=len(seen), wall_s=round(time.time() - w0, 1),
             maxrss_gb=round(ru.ru_maxrss / 2 ** 20, 2), exit=st, wall_killed=killed, t_end=time.time()))
    sys.exit(os.waitstatus_to_exitcode(st))
