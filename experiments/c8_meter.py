"""Campaign 08 CPU meter: run a command, append user+sys CPU of it and all its waited children (os.wait4;
includes threads) plus wall and peak RSS to runs/c8/cpu_ledger.jsonl. Limits: 1 BLAS/OMP thread,
prlimit --as 16 GB unless the command sets its own. usage: c7_meter.py KIND [--wall S] -- cmd args..."""
import json, os, signal, subprocess, sys, time
LEDGER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "c8", "cpu_ledger.jsonl")


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
    w0 = time.time(); p = subprocess.Popen(cmd, env=env); killed = False
    while True:
        pid, st, ru = os.wait4(p.pid, os.WNOHANG)
        if pid: break
        if time.time() - w0 > wall and not killed: os.kill(p.pid, signal.SIGKILL); killed = True
        time.sleep(0.5)
    p.returncode = st
    log(dict(kind=kind, cmd=" ".join(cmd)[:300], cpu_s=round(ru.ru_utime + ru.ru_stime, 1), wall_s=round(time.time() - w0, 1),
             maxrss_gb=round(ru.ru_maxrss / 2 ** 20, 2), exit=st, wall_killed=killed, t_end=time.time()))
    sys.exit(os.waitstatus_to_exitcode(st))
