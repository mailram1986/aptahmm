#!/usr/bin/env bash
# Reproduce every result reported in the README. Requires Python >= 3.10 and numpy.
#
# Copyright (c) 2026 Ram Kothandan,
# Centre for Biocomputing, Department of Biotechnology and
# Bioinformatics, Kumaraguru College of Technology, Coimbatore, India.
# Released under the MIT License. See LICENSE for details.
set -euo pipefail
cd "$(dirname "$0")/.."

cat <<'BANNER'
================================================================================
 aptahmm 1.0.0 - reproduction script

 Copyright (c) 2026 Ram Kothandan,
 Centre for Biocomputing, Department of Biotechnology and
 Bioinformatics, Kumaraguru College of Technology, Coimbatore, India.
 Released under the MIT License. See LICENSE for details.
================================================================================
BANNER
echo

SEEDS=data/seeds_ssdna_aptamers.fasta
OUT=results

mkdir -p "$OUT"

echo "== 1. seed-set inspection =="
python3 -m aptahmm.cli inspect --seeds "$SEEDS" | tee "$OUT/inspect.txt"

echo
echo "== 2. hidden-state model selection (includes the 1-state baseline) =="
python3 -m aptahmm.cli select-states --seeds "$SEEDS" \
    --min-states 1 --max-states 6 --em-iterations 500 --em-tolerance 1e-6 \
    --out "$OUT/model_selection.csv" | tee "$OUT/model_selection.txt"

echo
echo "== 3. per-sequence hidden-state occupancy (2-state model) =="
python3 scripts/state_occupancy.py "$SEEDS" --states 2 | tee "$OUT/state_occupancy.txt"

echo
echo "== 4. candidate generation, 23 nt, 2-state model =="
python3 -m aptahmm.cli generate --seeds "$SEEDS" \
    --length 23 --states 2 -n 200 --out "$OUT/run_k2" | tee "$OUT/generate_k2.txt"

echo
echo "== 5. candidate generation, 23 nt, 1-state (BIC-selected) baseline =="
python3 -m aptahmm.cli generate --seeds "$SEEDS" \
    --length 23 --states 1 -n 200 --out "$OUT/run_k1" | tee "$OUT/generate_k1.txt"

echo
echo "== 6. test suite =="
if python3 -c "import pytest" 2>/dev/null; then
    python3 -m pytest tests/ -q
else
    echo "SKIPPED: pytest is not installed, so the test suite was not run."
    echo "  The five steps above are complete and all reported results are in"
    echo "  $OUT/. To run the tests as well:"
    echo "      python3 -m pip install -r requirements-dev.txt"
    echo "  On Debian/Ubuntu/Fedora this requires a virtual environment"
    echo "  (PEP 668); see the Installation section of README.md."
fi

echo
echo "Done. All outputs are under $OUT/"
