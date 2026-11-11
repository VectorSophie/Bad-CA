"""Campaign 11 pipeline checks (CPU, no training): colour preprocessing, aspect ratio, cache separation, the Bad Apple
path, decoder validation, and export-only replay with the source unavailable. Run: .venv/bin/python tests/test_pipeline.py"""
import json, os, shutil, subprocess, sys, tempfile
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments"))
from ca import pipeline

COLOR = os.path.join(ROOT, "tests", "data", "color_ball_96x64.mp4"); BA = os.path.join(ROOT, "refs", "Bad-Apple-Starter", "BadApple!!.mp4")


def test_color_and_cache():
    d1, m1 = pipeline.prep(COLOR, 0, 24, 32, threshold=128); d2, m2 = pipeline.prep(COLOR, 0, 24, 48, threshold=128)
    d3, m3 = pipeline.prep(COLOR, 0, 24, 32, threshold=100); d4, _ = pipeline.prep(COLOR, 0, 24, 32, threshold=128)
    g1, b1 = np.load(os.path.join(d1, "gray.npy")), np.load(os.path.join(d1, "bin.npy"))
    assert g1.shape == (24, 21, 32) and np.load(os.path.join(d2, "bin.npy")).shape == (24, 32, 48)        # 96x64 -> aspect kept
    assert len({d1, d2, d3}) == 3 and d4 == d1                                                         # settings separate the cache; same settings hit it
    assert b1[0].any() and not b1[0].all() and (b1[0] != b1[-1]).any()                                 # ball is foreground and moves
    yy, xx = np.nonzero(b1[0]); assert g1[0][b1[0] == 1].min() >= 128                                   # luma >= threshold is foreground
    assert m1["preprocessing"]["luma"].startswith("PIL") and m1["source"]["sha256"] and m1["preprocessing"]["source_frame_index"] == list(range(24))
    _, m5 = pipeline.prep(COLOR, 0, 12, 32, fps=15, threshold=128)                                      # output fps: every other source frame
    assert m5["preprocessing"]["source_frame_index"] == list(range(0, 24, 2))


def test_bad_apple_path():
    from exp40_c11 import target
    d, m = pipeline.prep(BA, 4500, 56, 48, threshold=128)
    assert (np.load(os.path.join(d, "bin.npy")) == target("main")).all() and m["preprocessing"]["height"] == 36


def test_rejects():
    for bad in (os.path.join(ROOT, "ca", "pipeline.py"),):
        try: pipeline.probe(bad); raise AssertionError("non-video accepted")
        except ValueError as e: assert "decode" in str(e)
    try: pipeline.prep(COLOR, 40, 24, 32); raise AssertionError("interval past the end accepted")
    except ValueError as e: assert "lasts" in str(e)


def test_export_only_replay():
    """Copy an export to an empty temp dir and replay it there with Python (isolated -I, no repo data) and Node."""
    src = os.path.join(ROOT, "runs", "pipe", "example_c10_main", "model.ncab"); t = tempfile.mkdtemp(); shutil.copy(src, t)
    code = ("import sys; sys.path.insert(0, %r); import hashlib, numpy as np; from ca import ncab, nca; hd, P, f0 = ncab.read('model.ncab'); "
            "B, _ = nca.rollout_np(P, f0[None], hd['T'], hd['S']); print(hashlib.sha256(np.ascontiguousarray(B, np.uint8).tobytes()).hexdigest())") % ROOT
    py = subprocess.run([sys.executable, "-I", "-c", code], cwd=t, capture_output=True, text=True, check=True).stdout.strip()
    nd = json.loads(subprocess.run(["node", os.path.join(ROOT, "player", "nca_replay.mjs"), os.path.join(t, "model.ncab")], cwd=t, capture_output=True, text=True, check=True).stdout)
    assert py == nd["sha256"] == "f2401adb7b9faa1685a8e9c5e3a233c5fcfc297720eba6ae9f1db2a59b1670c4", (py, nd)
    shutil.rmtree(t)




def test_v2_info_import_replay():
    """ncab-v2 (phased) through the CLI functions: info, import (copied as is), export-only replay, comparison video."""
    v2 = os.path.join(ROOT, "runs", "featured", "interim_0-1467.ncab"); inf = pipeline.info(v2)
    assert inf["format"] == "ncab-v2" and inf["T"] == 1468 and inf["phases"] == 26
    run = tempfile.mkdtemp(); r, sizes = pipeline.import_export(v2, run, BA, 0, 48)
    assert sizes["format"] == "ncab-v2" and open(os.path.join(run, "model.ncab"), "rb").read() == open(v2, "rb").read()
    B, V, rep = pipeline.replay(os.path.join(run, "model.ncab"), node=False); assert B.shape == (1468, 36, 48)
    try: pipeline.load_export(v2); raise AssertionError("v1 loader accepted v2")
    except ValueError: pass
    shutil.rmtree(run)


def test_phased_training_cli():
    """--phases uniform:2 on the colour clip (CPU, tiny): chunked training -> ncab-v2 export -> Python == Node binary replay."""
    run = tempfile.mkdtemp(); env = dict(os.environ, JAX_PLATFORMS="cpu")
    subprocess.run([sys.executable, "-m", "ca.pipeline", "run", COLOR, run, "--frames", "24", "--width", "32", "--steps", "20", "--phases", "uniform:2", "--noise", "0.02"],
                   cwd=ROOT, env=env, check=True, capture_output=True)
    st = json.load(open(os.path.join(run, "status.json"))); assert st["status"] == "finished", st
    inf = pipeline.info(os.path.join(run, "model.ncab")); assert inf["format"] == "ncab-v2" and inf["K"] == 2 and inf["T"] == 24
    assert st["replay"]["binary_mismatch_px"] == 0, st["replay"]
    shutil.rmtree(run)


if __name__ == "__main__":
    for f in (test_color_and_cache, test_bad_apple_path, test_rejects, test_export_only_replay, test_v2_info_import_replay, test_phased_training_cli): f(); print("ok", f.__name__)
