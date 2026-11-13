"""Dataset registry and CLI error handling (cleanup pass).
1. BADCA_DATA relocates derived files: a copy of the data/ layout elsewhere loads identically; a root without the file
   raises FileNotFoundError that says how to obtain it; an unknown dataset raises KeyError listing the registered ones.
2. `python -m ca.datasets` lists both datasets.
3. `python -m ca.pipeline info <missing>` exits with code 2 and a one-line message instead of a traceback."""
import os, shutil, subprocess, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from ca import datasets

ref = datasets.frames("fantasmagorie"); rel = datasets.manifest("fantasmagorie")["derived"][datasets.PRIMARY["fantasmagorie"]]["file"]
with tempfile.TemporaryDirectory() as d:
    os.makedirs(os.path.join(d, os.path.dirname(rel))); shutil.copy(os.path.join(ROOT, rel), os.path.join(d, rel))
    os.environ["BADCA_DATA"] = d; assert (datasets.frames("fantasmagorie") == ref).all()
    os.environ["BADCA_DATA"] = os.path.join(d, "empty")
    try: datasets.frames("fantasmagorie"); raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError as e: assert "gen_data.py prep" in str(e)
    del os.environ["BADCA_DATA"]
try: datasets.frames("nope"); raise AssertionError("expected KeyError")
except KeyError as e: assert "fantasmagorie" in str(e)

out = subprocess.run([sys.executable, "-m", "ca.datasets"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
assert "bad_apple:" in out and "fantasmagorie:" in out
r = subprocess.run([sys.executable, "-m", "ca.pipeline", "info", "runs/does_not_exist.ncab"], cwd=ROOT, capture_output=True, text=True)
assert r.returncode == 2 and r.stderr.strip() == "error: file not found: runs/does_not_exist.ncab" and "Traceback" not in r.stderr
print("test_datasets_cli: all checks passed")
