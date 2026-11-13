set -e; cd /home/jackb/workspace/BadCA
[ -e runs/gen/fit/fa160_K16_96/train_done.json ] || XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python experiments/gen_train.py fit --out runs/gen/fit/fa160_K16_96 --ds fantasmagorie --rep binary_96x72_lines_tm3_v1 --start 160 --frames 240 --phases uniform:16 --steps 4000
[ -e runs/gen/fit/ba1760_K16_96/train_done.json ] || XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python experiments/gen_train.py fit --out runs/gen/fit/ba1760_K16_96 --ds bad_apple --rep binary_96x72_v1 --start 1760 --frames 240 --phases uniform:16 --steps 4000
