#!/usr/bin/env bash
# Reproduce the second study end to end, from a clean checkout.
#
#   ./run_all.sh            reproduce and verify against the paper
#   ./run_all.sh --figures  also regenerate the two PDF figures
#
# Creates a local virtualenv in ./venv on first run. Takes about a minute
# on 8 cores, most of it in the hyper-parameter grid search.
set -euo pipefail
cd "$(dirname "$0")"

PY=./venv/bin/python
if [ ! -x "$PY" ]; then
  echo "==> creating virtualenv with the pinned dependencies"
  python3 -m venv venv
  ./venv/bin/pip install --quiet --upgrade pip
  ./venv/bin/pip install --quiet -r requirements.txt
fi

echo "==> focus_study.py      (builds FocusDS, tunes and evaluates 7 algorithms)"
$PY focus_study.py

echo "==> focus_similarity.py (Similarity Score analysis, RQ5)"
$PY focus_similarity.py > /dev/null

if [ "${1:-}" = "--figures" ]; then
  echo "==> make_figs.py        (Figure 7 and the Similarity Score figure)"
  $PY make_figs.py
fi

echo "==> verify.py           (compare against the published numbers)"
$PY verify.py
