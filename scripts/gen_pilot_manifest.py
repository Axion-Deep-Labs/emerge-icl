"""Prints the Phase 0 pilot manifest, one cell per line: model_size pool_size seed.

Usage:
  python scripts/gen_pilot_manifest.py > scripts/pilot_manifest.txt
"""

MODEL_SIZES = ["small", "medium"]
POOL_SIZES = ["4", "16", "64", "256", "1024", "4096", "inf"]
SEEDS = [0, 1]

for size in MODEL_SIZES:
    for pool in POOL_SIZES:
        for seed in SEEDS:
            print(f"{size} {pool} {seed}")
