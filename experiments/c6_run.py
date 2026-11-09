"""Campaign 06 driver. PLAN.json = {"workers": N (<= 4; cube runs <= 2), "jobs": [
  {"target", "struct", "G", "variant", "solvers": [...], "cpu_s", "mem_gb", "outdir", "cubes": [optional list]}]}.
Each instance is encoded once (if its CNF is absent), then every solver (and cube) solves that same file
in its own process: prlimit --as/--cpu, wall cap 3x CPU cap (KILL). CPU = os.wait4 user+sys of the
process (includes its threads); every process is appended to runs/c6/cpu_ledger.jsonl. Resumable:
a (instance, solver, cube) with an existing ledger record is skipped.
Statuses: SAT / UNSAT / UNKNOWN_cpu_limit / UNKNOWN_wall_limit / UNKNOWN_memout / ERROR."""
import json, os, signal, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
from c6_meter import LEDGER, log
from exp19_csat import CNF_DIR, inst_name
STOP_CORE_H = float(os.environ.get("C6_STOP_CORE_H", "7.0"))
lock = threading.Lock(); enc_locks = {}


def core_h():
    return sum(json.loads(l)["cpu_s"] for l in open(LEDGER)) / 3600 if os.path.exists(LEDGER) else 0.0


def done(key):
    return os.path.exists(LEDGER) and any(json.loads(l).get("key") == key for l in open(LEDGER))


def run(args, cpu, mem, logpath, key, extra):
    cmd = ["prlimit", f"--as={int(mem * 2 ** 30)}", f"--cpu={int(cpu)}:{int(cpu) + 5}", sys.executable, os.path.join(HERE, "exp19_csat.py")] + args
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    w0 = time.time(); killed = False
    with open(logpath, "w") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        while True:
            pid, st, ru = os.wait4(p.pid, os.WNOHANG)
            if pid: break
            if time.time() - w0 > 3 * cpu and not killed: os.kill(p.pid, signal.SIGKILL); killed = True
            time.sleep(0.5)
    p.returncode = st
    cpu_s = ru.ru_utime + ru.ru_stime; txt = open(logpath).read()
    if st == 0: status = "OK"
    elif killed: status = "UNKNOWN_wall_limit"
    elif cpu_s >= cpu - 1: status = "UNKNOWN_cpu_limit"
    elif "MemoryError" in txt or "bad_alloc" in txt or "Cannot allocate" in txt: status = "UNKNOWN_memout"
    else: status = "ERROR"
    rec = dict(extra, key=key, cmd=" ".join(args), status=status, cpu_s=round(cpu_s, 1), wall_s=round(time.time() - w0, 1),
               maxrss_gb=round(ru.ru_maxrss / 2 ** 20, 2), cpu_cap_s=cpu, mem_gb=mem, exit=st, t_end=time.time())
    with lock: log(rec)
    print(json.dumps(rec), flush=True); return status


def job(j):
    name = inst_name(j["target"], j["struct"], j["G"], j["variant"])
    with lock: el = enc_locks.setdefault(name, threading.Lock())
    with el:
        if not os.path.exists(os.path.join(CNF_DIR, name + ".meta.json")):
            os.makedirs(CNF_DIR, exist_ok=True)
            if run(["encode", j["target"], j["struct"], str(j["G"]), j["variant"]], 1800, j.get("mem_gb", 24),
                   os.path.join(CNF_DIR, name + ".encode.log"), f"encode:{name}", dict(kind="encode", inst=name)) != "OK":
                return
    meta = json.load(open(os.path.join(CNF_DIR, name + ".meta.json")))
    for solver in j["solvers"]:
        for cube in j.get("cubes", [""]):
            key = f"solve:{name}:{solver}:{cube}"
            if done(key): continue
            if j.get("stop_on_sat") and os.path.exists(LEDGER) and any(
                    (r := json.loads(l)).get("kind") == "verdict" and r.get("verdict") == "SAT" and f":{name}:" in r["key"] for l in open(LEDGER)):
                print(json.dumps(dict(skipped=key, reason="instance already SAT in another cube")), flush=True); return
            if core_h() >= STOP_CORE_H:
                print(json.dumps(dict(skipped=key, reason="ledger stop rule")), flush=True); return
            out = os.path.join(ROOT, "runs", j["outdir"]); os.makedirs(out, exist_ok=True)
            stem = os.path.join(out, f"{name}__{solver}" + (f"__cube_{cube.replace(',', '_').replace('=', '')}" if cube else ""))
            st = run(["solve", name, solver, j["outdir"]] + ([cube] if cube else []), j["cpu_s"], j.get("mem_gb", 24), stem + ".log", key,
                     dict(kind="solve", inst=name, solver=solver, cube=cube or None, G=j["G"], variant=j["variant"], target=j["target"], struct=j["struct"],
                          cnf_sha256=meta["cnf_sha256"]))
            if st == "OK":
                r = json.load(open(stem + ".result.json"))
                with lock: log(dict(kind="verdict", key="verdict:" + key, cpu_s=0, verdict=r["status"], solve_cpu_s=r["solve_cpu_s"],
                                    exact=r.get("python_rollout_exact"), node=r.get("node_file_decode_exact"), file_bytes=r.get("file_bytes")))


if __name__ == "__main__":
    plan = json.load(open(sys.argv[1]))
    w0 = time.time()
    try:
        with ThreadPoolExecutor(min(4, plan.get("workers", 1))) as ex:
            list(ex.map(job, plan["jobs"]))
    finally:  # the driver's own CPU (its children are already logged per job)
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF)
        log(dict(kind="driver", key=f"driver:{sys.argv[1]}:{w0}", cpu_s=round(ru.ru_utime + ru.ru_stime, 1), wall_s=round(time.time() - w0, 1), t_end=time.time()))
