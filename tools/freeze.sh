#!/usr/bin/env bash
# Read-only, content-addressed snapshot of the code a job will run. Prints the snapshot dir.
set -euo pipefail
cd "$(dirname "$0")/.."
h=$(find ca experiments player tests -type f \( -name '*.py' -o -name '*.mjs' \) -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-12)
d="runs/frozen/$h"
if [ ! -d "$d" ]; then
  mkdir -p "$d"
  cp -r ca experiments player tests "$d"/
  find "$d" -name __pycache__ -prune -exec rm -rf {} +
  ln -s ../../../refs "$d/refs"; ln -s ../../../cache "$d/cache"; ln -s ../.. "$d/runs"
  chmod -R a-w "$d/ca" "$d/experiments" "$d/player" "$d/tests"
fi
echo "$PWD/$d"
