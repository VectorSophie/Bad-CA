#!/bin/bash
# usage: d1_jobs.sh PART (0|1): runs every (task, objective, seed) whose index % 2 == PART
cd /home/jackb/workspace/BadCA; i=0
for t in gh translate blink majority propagate; do for o in onestep rollout curriculum; do for s in 0 1 2; do
  if [ $((i % 2)) -eq $1 ]; then f=""; [ "$t" = gh ] && [ $s = 0 ] && f=--force
    .venv/bin/python experiments/pd_d1.py $t $o $s 32000 $f > runs/path_d/d1/${t}_${o}_s${s}.log 2>&1; fi; i=$((i+1)); done; done; done
