#!/bin/sh
# Rebuild the ready-to-play examples in runs/pipe/example_* (exports are committed; the original comparison frames live
# in the git-ignored cache/pipe and are regenerated here from the source video). No training.
set -e; cd "$(dirname "$0")/.."; PY=.venv/bin/python; BA="refs/Bad-Apple-Starter/BadApple!!.mp4"
imp() { $PY -m ca.pipeline import "$1" "runs/pipe/$2" --video "$BA" --start "$3" --width 48 --label "$4"; }
imp runs/c10/exports/c10_combo10_main_best_BC_s0.nca.json example_c10_main 4500 "C10 B+C s0 (10,320 params), Bad Apple 4500-4555"
imp runs/c10/exports/c10_combo10_confirm_best_BC_s1.nca.json example_c10_confirm 1800 "C10 B+C s1 (10,320 params), Bad Apple 1800-1859"
imp runs/c11/exports/p1_main_12x32_D_D_s0.ncab example_c11_main_3884 4500 "C11 small 3,884 params (direct, rep s0), Bad Apple 4500-4555 — C9-level, below C10"
imp runs/c11/exports/L120_big16x64_r2_s0.ncab example_c11_L120_10320 1760 "C11 long 120 frames, 10,320 params (s0), Bad Apple 1760-1879"
imp runs/c11/exports/L120_small12x32_s0.ncab example_c11_L120_3884 1760 "C11 long 120 frames, 3,884 params (s0) — fails sustained-motion gate"
imp runs/c11/exports/L240_big16x64_scratch_s0.ncab example_c11_L240_10320_fails 1760 "C11 240 frames, 10,320 params, 1 seed — EXPLORATORY, fails sustained-motion gate (contains scene changes)"
$PY -m ca.pipeline prep tests/data/color_ball_96x64.mp4 --frames 24 --width 32 --threshold 128   # colour demo's comparison frames
