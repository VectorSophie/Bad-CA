#!/bin/sh
# Snapshot ca/ experiments/ player/ tests/ into read-only runs/frozen/<sha12 of their contents>; print the dir.
# Runs are launched from the snapshot so later edits never change a running job (same layout as earlier campaigns).
set -e; cd "$(dirname "$0")/.."
h=$(find ca experiments player tests -type f ! -path '*__pycache__*' | sort | xargs sha256sum | sha256sum | cut -c1-12)
d=runs/frozen/$h
if [ ! -d "$d" ]; then
  mkdir -p "$d"; for x in ca experiments player tests; do mkdir "$d/$x"; find $x -maxdepth 1 -type f -exec cp {} "$d/$x/" \;; done
  ln -s ../../../cache "$d/cache"; ln -s ../../../refs "$d/refs"; ln -s ../.. "$d/runs"; chmod -R a-w "$d/ca" "$d/experiments" "$d/player" "$d/tests"
fi
echo "$PWD/$d"
