#!/bin/bash
# EMERGE Phase 0B measurement validation, for NMSU Discovery.
#
# This stage trains nothing. Every predictor in the battery is closed form, so
# it is CPU work of a few minutes and asks for no GPU. Cells that already have
# a result file are skipped, so the job is safe to resubmit.
#
# Submit (from the repo root on the cluster):
#   EMERGE_DIR=$PWD EMERGE_ENV=/fs1/scratch/$USER/emerge-env \
#     sbatch scripts/slurm/emerge_phase0b.slurm
#
#SBATCH --job-name=emerge-p0b
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

PROJECT_DIR="${EMERGE_DIR:-$HOME/emerge-icl}"
VENV="${EMERGE_ENV:-/fs1/scratch/$USER/emerge-env}"

cd "$PROJECT_DIR"
echo "node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# The emerge package sits at the repo root and there is no installed
# distribution, so a script run as scripts/run_phase0b.py puts scripts/ on
# sys.path and not the root. pytest finds the package on its own; plain python
# does not.
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# The measure tests are a gate, not the deliverable. If pytest is absent from
# the environment, say so and run the battery anyway rather than aborting.
if "$VENV/bin/python" -c "import pytest" 2>/dev/null; then
  "$VENV/bin/python" -m pytest tests/test_phase0b.py -q
else
  echo "WARNING: pytest not installed in $VENV, skipping the measure tests"
fi

"$VENV/bin/python" scripts/run_phase0b.py --config configs/phase0b.yaml --out results/phase0b
"$VENV/bin/python" scripts/check_phase0b_criteria.py --config configs/phase0b.yaml --results results/phase0b
