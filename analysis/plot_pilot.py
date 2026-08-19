"""Phase 0 pilot figures from results/pilot/*/metrics.jsonl.

Usage:
  python analysis/plot_pilot.py --results results/pilot --out analysis/figures

Produces:
  alignment_vs_M.png    final alignment score A versus pool size, per model size
  alignment_vs_step.png A over training steps, one panel per model size
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_runs(results_dir: Path):
    runs = []
    for metrics in sorted(results_dir.glob("*/metrics.jsonl")):
        cfg = json.loads((metrics.parent / "config.json").read_text())
        records = [json.loads(line) for line in metrics.read_text().splitlines()]
        runs.append((cfg, records))
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/pilot")
    ap.add_argument("--out", default="analysis/figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(Path(args.results))
    if not runs:
        raise SystemExit(f"no runs found under {args.results}")

    # Final A versus M, per model size (finite pools only; A is undefined at inf).
    final = defaultdict(list)
    for cfg, records in runs:
        if cfg["pool_size"] == "inf":
            continue
        a = records[-1].get("alignment_A")
        if a is not None:
            final[cfg["model_size"]].append((int(cfg["pool_size"]), cfg["seed"], a))

    fig, ax = plt.subplots(figsize=(7, 5))
    for size, points in sorted(final.items()):
        by_m = defaultdict(list)
        for m, _, a in points:
            by_m[m].append(a)
        ms = sorted(by_m)
        means = [sum(by_m[m]) / len(by_m[m]) for m in ms]
        ax.plot(ms, means, marker="o", label=size)
        for m in ms:
            ax.scatter([m] * len(by_m[m]), by_m[m], s=12, alpha=0.4)
    ax.axhline(0.25, ls=":", c="gray", label="emergence cutoff (A=0.25)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("pretraining task pool size M")
    ax.set_ylabel("alignment A (1 = dMMSE-like, 0 = ridge-like)")
    ax.set_title("EMERGE pilot: in-context emergence versus task diversity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "alignment_vs_M.png", dpi=160)

    # A over training, one panel per model size.
    sizes = sorted({cfg["model_size"] for cfg, _ in runs})
    fig, axes = plt.subplots(1, len(sizes), figsize=(6 * len(sizes), 5), squeeze=False)
    for ax, size in zip(axes[0], sizes):
        for cfg, records in runs:
            if cfg["model_size"] != size or cfg["pool_size"] == "inf":
                continue
            steps = [r["step"] for r in records if r.get("alignment_A") is not None]
            a = [r["alignment_A"] for r in records if r.get("alignment_A") is not None]
            ax.plot(steps, a, alpha=0.7, label=f"M={cfg['pool_size']} s{cfg['seed']}")
        ax.axhline(0.25, ls=":", c="gray")
        ax.set_xscale("symlog")
        ax.set_xlabel("training step")
        ax.set_ylabel("alignment A")
        ax.set_title(size)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "alignment_vs_step.png", dpi=160)
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
