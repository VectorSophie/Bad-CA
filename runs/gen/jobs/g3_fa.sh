set -e; cd /home/jackb/workspace/BadCA
[ -e runs/gen/fit/fa160_K1/train_done.json ] || XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python experiments/gen_train.py fit --out runs/gen/fit/fa160_K1 --ds fantasmagorie --start 160 --frames 240 --phases uniform:1 --steps 4000
[ -e runs/gen/fit/fa160_K4/train_done.json ] || XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python experiments/gen_train.py fit --out runs/gen/fit/fa160_K4 --ds fantasmagorie --start 160 --frames 240 --phases uniform:4 --steps 4000
[ -e runs/gen/fit/fa160_K16/train_done.json ] || XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python experiments/gen_train.py fit --out runs/gen/fit/fa160_K16 --ds fantasmagorie --start 160 --frames 240 --phases uniform:16 --steps 4000
[ -e runs/gen/fit/fa160_K1_Fh256/train_done.json ] || XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python experiments/gen_train.py fit --out runs/gen/fit/fa160_K1_Fh256 --ds fantasmagorie --start 160 --frames 240 --phases uniform:1 --steps 4000 --Fh 256
