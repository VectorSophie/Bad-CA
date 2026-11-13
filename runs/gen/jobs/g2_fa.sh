set -e; cd /home/jackb/workspace/BadCA
[ -e runs/gen/fit/fa_full/train_done.json ] || XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python experiments/gen_train.py fit --out runs/gen/fit/fa_full --ds fantasmagorie --start 0 --frames 1687 --phases auto:28 --steps 8000 --max-hours 2.5
