set -e; cd /home/jackb/workspace/BadCA
[ -e runs/gen/fit/ba1760_K1/train_done.json ] || XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python experiments/gen_train.py fit --out runs/gen/fit/ba1760_K1 --ds bad_apple --start 1760 --frames 240 --phases uniform:1 --steps 4000
[ -e runs/gen/fit/ba1760_K4/train_done.json ] || XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python experiments/gen_train.py fit --out runs/gen/fit/ba1760_K4 --ds bad_apple --start 1760 --frames 240 --phases uniform:4 --steps 4000
[ -e runs/gen/fit/ba1760_K16/train_done.json ] || XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python experiments/gen_train.py fit --out runs/gen/fit/ba1760_K16 --ds bad_apple --start 1760 --frames 240 --phases uniform:16 --steps 4000
[ -e runs/gen/fit/ba1760_K1_Fh256/train_done.json ] || XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python experiments/gen_train.py fit --out runs/gen/fit/ba1760_K1_Fh256 --ds bad_apple --start 1760 --frames 240 --phases uniform:1 --steps 4000 --Fh 256
