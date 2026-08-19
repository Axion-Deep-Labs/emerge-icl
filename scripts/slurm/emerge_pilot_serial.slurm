#!/bin/bash
# EMERGE Phase 0 pilot, serial runner for NMSU Discovery.
#
# Discovery's QOS caps submitted jobs per user, and pilot cells take only
# minutes each on an A100, so one serial job through the whole manifest is
# simpler and friendlier than a wide array. Completed cells are skipped, so
# the job is safe to resubmit if it is interrupted.
#
# Setup (once, from the repo root on the cluster):
#   $EMERGE_ENV/bin/python scripts/gen_pilot_manifest.py > scripts/pilot_manifest.txt
# Submit (from the repo root):
#   EMERGE_DIR=$PWD EMERGE_ENV=/path/to/venv sbatch scripts/slurm/emerge_pilot_serial.slurm
#
#SBATCH --job-name=emerge-pilot
#SBATCH --partition=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

PROJECT_DIR="${EMERGE_DIR:-$HOME/emerge-icl}"
VENV="${EMERGE_ENV:-/fs1/scratch/$USER/emerge-env}"
MANIFEST="$PROJECT_DIR/scripts/pilot_manifest.txt"

cd "$PROJECT_DIR"
TOTAL_STEPS=$(awk '/^train_steps:/ {print $2}' configs/pilot.yaml)
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"

while read -r MODEL_SIZE POOL_SIZE SEED; do
  RUN="results/pilot/${MODEL_SIZE}_M${POOL_SIZE}_s${SEED}"
  if [ -f "$RUN/metrics.jsonl" ] && grep -q "\"step\": ${TOTAL_STEPS}," "$RUN/metrics.jsonl"; then
    echo "skip (complete): $RUN"
    continue
  fi
  echo "[$(date '+%F %T')] run: model=$MODEL_SIZE pool=$POOL_SIZE seed=$SEED"
  "$VENV/bin/python" -m emerge.train \
    --config configs/pilot.yaml \
    --model-size "$MODEL_SIZE" \
    --pool-size "$POOL_SIZE" \
    --seed "$SEED" \
    --out results/pilot
done < "$MANIFEST"

echo "[$(date '+%F %T')] pilot manifest complete"
