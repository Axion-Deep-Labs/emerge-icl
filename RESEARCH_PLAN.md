# EMERGE Research Plan (Pre-Registered)

**Project:** EMERGE (Empirical Mapping of Emergent Regimes in GEneralization)
**Organization:** Axion Deep Labs, Inc.
**Status:** Pre-registered design, written before any pilot or full run. Amendments must be logged in EXPERIMENT_LOG.md with dates and reasons.
**Date:** 2026-08-19

---

## 1. Research Question

When a transformer is trained from scratch on a distribution of tasks, under what conditions does true in-context learning emerge, and how does the emergence threshold scale with pretraining task diversity, model capacity, and training compute?

We use the in-context linear regression setting because it admits exact optimal baselines. A model that has memorized its pretraining task pool behaves like the discrete Bayesian predictor over that pool (dMMSE). A model that has learned the general-purpose algorithm behaves like ridge regression under the continuous task prior. The distance between the model and these two references cleanly separates in-weights learning from in-context learning.

## 2. Hypotheses

- **H1 (diversity threshold):** For a fixed model, there exists a task diversity threshold M* such that models pretrained on M < M* tasks track the dMMSE predictor on unseen tasks, and models pretrained on M > M* tasks track ridge regression. This replicates prior reports in the literature and is our sanity gate, not our contribution.
- **H2 (capacity scaling, primary):** M* increases with model capacity. Larger models can afford to memorize larger pools, so they need more diversity before the general algorithm becomes the cheaper solution. We pre-register the direction (M* grows with parameter count) and will fit the functional form exploratorily.
- **H3 (onset dynamics):** Along training time at fixed M near the threshold, the transition from dMMSE-like to ridge-like behavior is sharp in training steps (an emergence event) rather than gradual, as measured by the sharpness statistic defined in section 5.
- **H4 (transience):** For M slightly above M*, in-context ability can peak and then decay with continued training as the model slowly memorizes the pool. We pre-register this as exploratory.

## 3. Setting

- **Task family:** noisy linear regression in d dimensions. A task is a weight vector w drawn from N(0, I_d). Inputs x ~ N(0, I_d), targets y = w.x + noise, noise ~ N(0, sigma^2).
- **Prompt format:** the sequence (x_1, y_1, ..., x_k, y_k, x_query). The model predicts y at every x position. Loss is mean squared error at x positions.
- **Task diversity M:** pretraining samples tasks from a fixed pool of M weight vectors (drawn once per run seed). Evaluation uses (a) pool tasks and (b) fresh tasks never seen in training.
- **Model:** decoder-only transformer trained from scratch. No pretrained weights of any kind.
- **Data:** fully synthetic, generated on the fly from seeded RNGs. No external datasets.

## 4. Baselines (computed in closed form, no training)

- **Ridge regression:** the Bayes-optimal predictor under the continuous prior w ~ N(0, I), computed per prompt prefix with the known noise level.
- **dMMSE:** the Bayes-optimal predictor under the discrete uniform prior over the run's M pool tasks (posterior-weighted mixture with Gaussian likelihood).
- **Zero predictor:** predicts 0, anchors the scale.

## 5. Endpoints

- **Primary endpoint:** the normalized alignment score A = (L_model - L_ridge) / (L_dmmse - L_ridge) on unseen tasks, averaged over context positions k in a pre-declared window. A near 1 means in-weights behavior, A near 0 means in-context behavior. **Emergence is declared when A <= 0.25 on unseen tasks.** The threshold M* for a configuration is the smallest M in the sweep meeting this at the end of training.
- **Secondary endpoints:** (a) seen-vs-unseen loss gap; (b) onset step, the first training step where A <= 0.25 holds and continues to hold for the remainder of training; (c) sharpness, the ratio of the A range crossed in the fastest 5 percent of training steps around onset to the total A range; (d) transience depth, max drop in unseen-task performance after its best point.
- The endpoint hierarchy is frozen. Secondary results will not be promoted to primary claims.

## 6. Design

### Phase 0, Pilot (local RTX 4090 or single Discovery A100)
- d = 8, context length 16 examples, sigma = 0.25.
- Model grid: 2 sizes (about 0.3M and 3M parameters).
- Diversity grid: M in {2^2, 2^4, 2^6, 2^8, 2^10, 2^12, infinity} where infinity means fresh tasks every batch.
- 2 seeds. Roughly 28 short runs.
- **Pilot gate:** H1 must replicate (both model sizes show A near 1 at the lowest M and A <= 0.25 at M = infinity on unseen tasks). If the gate fails we debug the setup, we do not proceed to Phase 1.

### Phase 1, Full grid (NMSU Discovery, SLURM array)
- Model grid: 5 to 6 sizes spanning roughly 0.1M to 30M parameters (width and depth scaled together).
- Diversity grid: 10 to 12 log-spaced M values, refined around each size's threshold region after a first pass.
- 3 seeds minimum per cell; 5 near thresholds.
- Dense checkpointing (log-spaced steps) so onset and transience are resolved.

### Phase 2, Robustness (exploratory, pre-declared as such)
- Vary d, noise level, and context length at a subset of cells to test that the H2 scaling survives.

## 7. Analysis Plan

- H2 is tested by Spearman correlation between parameter count and M* across the model grid, with a pre-declared significance level of 0.05. With 5 to 6 model sizes this is a coarse test; the paper will report effect sizes and per-seed thresholds, not just the p-value.
- Fits of M* versus parameters (power law versus log) are exploratory model selection, reported with both candidates.
- All runs are reproducible from config plus seed. Every figure in the paper must regenerate from committed analysis scripts plus released result files.

## 8. What Could Invalidate This

- Threshold M* not identifiable inside a feasible M sweep for larger models (compute ceiling). Mitigation: cap the largest model so its expected threshold sits inside the sweep, based on pilot extrapolation.
- The A <= 0.25 cutoff is arbitrary. Mitigation: report threshold curves at 0.1, 0.25, 0.5 as a sensitivity analysis; the pre-registered claim uses 0.25.
- Optimizer or curriculum artifacts. Mitigation: fixed AdamW recipe across all cells, no per-cell tuning.

## 9. Open Source and Publication

- MIT license from the first commit. Fully synthetic data, from-scratch models, no third-party weights, datasets, or model claims.
- Target: arXiv preprint, then TMLR or a NeurIPS/ICLR workshop.
- Result files (metrics JSONL, not checkpoints) are released with the paper.
