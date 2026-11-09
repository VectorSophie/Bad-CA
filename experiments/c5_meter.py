"""Run one command and append its CPU (user+sys incl. children, via os.wait4) to the Campaign 05 ledger.
usage: c5_meter.py KIND -- cmd args..."""
import json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c5_run import log

kind = sys.argv[1]; cmd = sys.argv[sys.argv.index("--") + 1:]
w0 = time.time(); p = subprocess.Popen(cmd); _, rc, ru = os.wait4(p.pid, 0)
log(dict(kind=kind, cmd=" ".join(cmd)[:300], cpu_s=round(ru.ru_utime + ru.ru_stime, 1), wall_s=round(time.time() - w0, 1),
         maxrss_gb=round(ru.ru_maxrss / 2 ** 20, 2), rc=rc, t_end=time.time()))
sys.exit(rc >> 8 if rc > 255 else rc)
