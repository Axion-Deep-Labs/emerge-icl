"""Run the Phase 0B validation battery and write per-cell results.

Trains nothing. Every predictor is closed form, so this is CPU work and takes
minutes. See PHASE_0B_PLAN.md for the design and the five pass criteria.

Usage:
  python scripts/run_phase0b.py --config configs/phase0b.yaml --out results/phase0b
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import torch
import yaml

from emerge.phase0b import battery_predictions, build_probes, relative_loss, score
from emerge.tasks import TaskSampler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase0b.yaml")
    ap.add_argument("--out", default="results/phase0b")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    sigma = cfg["sigma"]
    tau = cfg["tau_sigma_multiple"] * sigma
    assert cfg["query_norm"] == "sqrt_dim", "only the frozen query norm is implemented"

    for pool_size in cfg["pool_sizes"]:
        for seed in cfg["seeds"]:
            sampler = TaskSampler(cfg["dim"], sigma, pool_size, seed)
            for family in cfg["families"]:
                for k in cfg["context_lengths"]:
                    name = f"M{pool_size}_s{seed}_{family}_k{k}"
                    dest = out / f"{name}.json"
                    if dest.exists():
                        print(f"skip {name}")
                        continue
                    # Probe and predictor randomness is derived from the cell so
                    # a rerun of any cell reproduces exactly.
                    g = torch.Generator().manual_seed(
                        zlib.crc32(name.encode()) & 0x7FFFFFFF
                    )
                    probe = build_probes(
                        sampler, cfg["n_probes"], k, family, sigma, g
                    )
                    preds = battery_predictions(probe, sampler, sigma, cfg, g)
                    rows = {}
                    for pname, p in preds.items():
                        rows[pname] = {
                            **score(p, probe["pred_ridge"], probe["pred_dmmse"], tau),
                            "R": relative_loss(probe, p),
                        }
                    rec = {
                        "pool_size": pool_size,
                        "seed": seed,
                        "family": family,
                        "context_len": k,
                        "tau": tau,
                        "n_probes": cfg["n_probes"],
                        "divergence_mean": float(
                            (probe["pred_dmmse"] - probe["pred_ridge"]).abs().mean()
                        ),
                        "predictors": rows,
                    }
                    dest.write_text(json.dumps(rec, indent=2))
                    ok = rows["perfect_ridge"]
                    print(
                        f"{name}: admitted {ok['n_admitted']}/{cfg['n_probes']} "
                        f"({ok['admit_rate']:.1%})  ridge S={ok['S']:+.4f} E={ok['E']:.4f}  "
                        f"dmmse S={rows['perfect_dmmse']['S']:+.4f}"
                    )


if __name__ == "__main__":
    main()
