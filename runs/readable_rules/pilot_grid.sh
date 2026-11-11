#!/bin/sh
# Readable Rules pilot grid (CPU): paths 'extend simple rules' (library) and 'direct' (tree) on two diagnostic tasks
cd "$(dirname "$0")/../.."; export CA_LEDGER=$PWD/runs/readable_rules/cpu_ledger.jsonl
R=".venv/bin/python experiments/c9_meter.py"
for task in 4500:56 1221:60; do t=$(echo $task | tr ':' '_')
 for k in 0 1; do for K in 1 4 8; do for g in 0 0.1; do
  $R rr_lib -- .venv/bin/python experiments/rr_run.py $task pilot/${t}_lib_k${k}_S2_K${K}_g${g} --start library --k $k --S 2 --K $K --gamma $g --evals 2000 --seeds 0,1,2 --max-rules 8 --max-conds 3 2>&1 | grep "s[012]:"
 done; done; done
 for K in 1 4 8; do for g in 0 0.1; do
  $R rr_tree -- .venv/bin/python experiments/rr_run.py $task pilot/${t}_tree_k0_S1_K${K}_g${g} --start tree --k 0 --S 1 --K $K --gamma $g --evals 2000 --seeds 0,1,2 --max-rules 32 --max-conds 4 2>&1 | grep "s[012]:"
 done; done
done
