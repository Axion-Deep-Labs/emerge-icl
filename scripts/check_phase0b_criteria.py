"""Score the five Phase 0B pass criteria against results/phase0b.

The measure is adopted only if all five hold. See PHASE_0B_PLAN.md section 5.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def load(results: Path):
    return [json.loads(f.read_text()) for f in sorted(results.glob("*.json"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase0b.yaml")
    ap.add_argument("--results", default="results/phase0b")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    recs = load(Path(args.results))
    if not recs:
        raise SystemExit("no results found; run scripts/run_phase0b.py first")

    primary = [r for r in recs if r["context_len"] == cfg["primary_context_length"]]
    e_max, ridge_band = cfg["e_max"], cfg["s_ridge_band"]
    verdicts = {}

    # 1. Incompetence cannot look algorithmic.
    bad = [
        (r["pool_size"], r["seed"], r["family"])
        for r in recs
        if r["predictors"]["null"]["S"] <= ridge_band
        and r["predictors"]["null"]["E"] <= e_max
    ]
    verdicts["1_null_not_algorithmic"] = (not bad, f"{len(bad)} cells read the null predictor as ridge-like")

    # 2. Interpretation stable across pool size.
    drift = []
    for pname, target in (("perfect_ridge", 0.0), ("perfect_dmmse", 1.0)):
        vals = [r["predictors"][pname]["S"] for r in primary]
        drift.append((pname, min(vals), max(vals), max(abs(v - target) for v in vals)))
    worst = max(d[3] for d in drift)
    verdicts["2_stable_across_M"] = (worst <= cfg["tol_stability"], f"worst deviation {worst:.4f} vs tol {cfg['tol_stability']}")

    # 3. Mixtures recover alpha monotonically.
    alphas = cfg["mixture_alphas"]
    worst_mix, non_monotone = 0.0, 0
    for r in primary:
        s = [r["predictors"][f"mix_{a}"]["S"] for a in alphas]
        worst_mix = max(worst_mix, max(abs(si - a) for si, a in zip(s, alphas)))
        if any(s[i + 1] < s[i] for i in range(len(s) - 1)):
            non_monotone += 1
    verdicts["3_mixtures_recover"] = (
        worst_mix <= cfg["tol_mixture"] and non_monotone == 0,
        f"worst |S-alpha| {worst_mix:.4f} vs tol {cfg['tol_mixture']}, {non_monotone} non-monotone cells",
    )

    # 4. Output noise does not change the inferred mechanism.
    worst_noise = 0.0
    for r in primary:
        for base, ref in (("ridge", "perfect_ridge"), ("dmmse", "perfect_dmmse")):
            s0 = r["predictors"][ref]["S"]
            for lv in cfg["noise_levels"]:
                worst_noise = max(worst_noise, abs(r["predictors"][f"noisy_{base}_{lv}"]["S"] - s0))
    verdicts["4_noise_invariant"] = (worst_noise <= cfg["tol_noise"], f"worst S shift under noise {worst_noise:.4f} vs tol {cfg['tol_noise']}")

    # 5. S is not a function of R. Bucket R and require S to span within a bucket.
    pairs = [
        (p["R"], p["S"])
        for r in primary
        for p in r["predictors"].values()
        if p["n_admitted"] > 0
    ]
    spread = 0.0
    for lo, hi in ((0.0, 1.5), (1.5, 3.0), (3.0, 1e9)):
        s = [sv for rv, sv in pairs if lo <= rv < hi]
        if len(s) >= 2:
            spread = max(spread, max(s) - min(s))
    verdicts["5_not_a_transform_of_loss"] = (spread >= 0.5, f"largest S spread within an R bucket {spread:.4f}")

    width = max(len(k) for k in verdicts)
    all_pass = True
    for k, (ok, detail) in verdicts.items():
        all_pass &= ok
        print(f"{k:<{width}}  {'PASS' if ok else 'FAIL'}  {detail}")
    print()
    print("ADOPT the measure" if all_pass else "DO NOT ADOPT: at least one criterion failed")


if __name__ == "__main__":
    main()
