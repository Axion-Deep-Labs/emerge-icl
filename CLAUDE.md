# EMERGE (Empirical Mapping of Emergent Regimes in GEneralization)

Open source research program by Axion Deep Labs. Public repo:
https://github.com/Axion-Deep-Labs/emerge-icl (MIT).

## Ground rules for this repo

1. **The repo is public.** Never commit credentials, HPC usernames in configs,
   internal business context, or anything from ~/Corporate outside this folder.
2. **RESEARCH_PLAN.md is pre-registered and frozen.** Amendments go in
   EXPERIMENT_LOG.md with a date and reason; never silently edit the plan.
3. **Zero legal exposure by design:** all data is synthetic from seeded RNGs,
   all models are trained from scratch, no third-party weights or datasets,
   no claims about anyone's commercial product. Keep it that way.
4. **Never commit results/ or data/.** Result files are released as artifacts
   with the paper.
5. Work happens on branches and merges by PR; do not push straight to main.
6. Long-running training is launched by Joshua (locally or via sbatch on
   Discovery), not by Claude. Provide the command instead.
7. Every figure must regenerate from a committed script in analysis/ plus
   released result files.

## Quick reference

- Train one cell:
  `.venv/bin/python -m emerge.train --config configs/pilot.yaml --model-size small --pool-size 16 --seed 0`
- Tests: `.venv/bin/python -m pytest tests/ -q`
- Pilot on Discovery: generate `scripts/pilot_manifest.txt`, then
  `sbatch --array=1-N scripts/slurm/emerge_pilot.slurm` (N = manifest lines).
  Env vars: EMERGE_DIR (repo path on cluster), EMERGE_ENV (venv path).
- Figures: `python analysis/plot_pilot.py --results results/pilot`

## Science summary

Setting: in-context noisy linear regression, prompts of (x, y) pairs.
Pretraining draws tasks from a pool of M weight vectors. Two closed-form
references bracket the model's behavior on unseen tasks: dMMSE (Bayes over
the discrete pool, in-weights behavior) and ridge regression (Bayes over the
continuous prior, in-context behavior). Alignment score
A = (L_model - L_ridge) / (L_dmmse - L_ridge); emergence means A <= 0.25.
Primary hypothesis (H2): the diversity threshold M* grows with model
capacity. Full design in RESEARCH_PLAN.md.
