#!/bin/bash
# EMERGE Phase 0 pilot on NMSU Discovery. One array task per grid cell.
#
# Setup (once):
#   python scripts/gen_pilot_manifest.py > scripts/pilot_manifest.txt
# Submit (array size = line count of the manifest):
#   sbatch --array=1-$(wc -l < scripts/pilot_manifest.txt) scripts/slurm/emerge_pilot.slurm
#
#SBATCH --job-name=emerge-pilot
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail

PROJECT_DIR="${EMERGE_DIR:-$HOME/emerge-icl}"
VENV="${EMERGE_ENV:-/fs1/scratch/$USER/emerge-env}"
MANIFEST="$PROJECT_DIR/scripts/pilot_manifest.txt"

cd "$PROJECT_DIR"
read -r MODEL_SIZE POOL_SIZE SEED < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")

echo "cell: model=$MODEL_SIZE pool=$POOL_SIZE seed=$SEED node=$(hostname)"
"$VENV/bin/python" -m emerge.train \
  --config configs/pilot.yaml \
  --model-size "$MODEL_SIZE" \
  --pool-size "$POOL_SIZE" \
  --seed "$SEED" \
  --out "results/pilot"
