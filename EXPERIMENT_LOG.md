# EMERGE Experiment Log

Dated log of every run batch, decision, and amendment to the pre-registered
plan. Newest entries at the top.

---

## 2026-09-08: Phase 0 complete, endpoint invalidated, Phase 1 paused

**Runs.** Phase 0 pilot finished at 28 of 28 cells (2 model sizes, 7 pool sizes,
2 seeds), 102 evaluations per run, 50,000 steps per run. Results are in
`results/pilot/`.

**Deviations from the plan, logged now rather than at the time.**

- The plan specified model sizes of roughly 0.3M and 3M parameters. The runs used
  152,833 and 4,750,081 parameters.
- A100 nodes were unavailable, so the pilot ran on interactive-partition MIG slices
  rather than the allocation named in the plan.
- `save_final_checkpoint` was false, so no model weights were retained from any cell.

**Finding: the pre-registered alignment score carries no information beyond loss.**

With `R = L_model / L_ridge` and `G = (L_dmmse - L_ridge) / L_ridge`, the score
defined in RESEARCH_PLAN.md section 5 satisfies the exact identity

    A = (R - 1) / G

verified in the logged data to 4.4e-16 across all 2,448 evaluations. G is computed
from the two closed-form references on the evaluation sample and is bit-for-bit
constant across all 102 evaluations within every cell.

Two consequences.

1. A fixed cutoff on A is an accuracy requirement that slides with pool size. At
   `A <= 0.25` the permitted relative loss was 8.52 at M=4, 5.87 at M=64 and 3.15 at
   M=4096 (seed 0). It is not a comparable criterion across the sweep.
2. Within a cell, A and R are the same curve under an affine map, so any behaviour
   described in one is the same behaviour described in the other.

**Scoring outcomes.**

- **H1 gate: invalid as pre-registered, not passed.** H1 was operationalized through
  the same cutoff. What survives is descriptive: relative loss on unseen tasks improves
  as the pool grows, in both sizes and both seeds.
- **H2: unscored.** Under a competence criterion `R <= 1.25`, the small model reaches
  no threshold at any tested diversity, its best relative loss anywhere being 2.21. The
  medium model gives `256 < M* <= 1024`. One qualifying model size cannot test a
  scaling hypothesis, so H2 is neither supported nor refuted.
- The small model passed the pre-registered cutoff at M = 64, 256, 1024 and 4096 while
  never coming closer than 2.21 times ridge loss. That is the pilot's substantive
  result and is what a Phase 0 exists to find.

**Competence readings, for the record.** Qualifying windows require `R <= 1.25` held
over at least 2,000 optimizer steps and at least three evaluations; evaluation spacing
in this pilot ranges from 1 step to 1,000, so persistence counted in evaluations is not
a duration. Windows: medium M=1024 seed 0 from step 17,783, worst R 1.248813; seed 1
from 32,000, worst R 1.249928; medium M=4096 seed 0 from 13,000, worst R 1.240247;
seed 1 from 17,783, worst R 1.246018. All run to step 50,000. At infinite diversity
over the final window, medium reads 1.022 and 1.046, small reads 2.967 and 2.253.

The 1.25 bound is an operational tolerance chosen after seeing the pilot. It is not
derived. Sensitivity: at 1.50 and 1.25 the medium bracket is `256 < M* <= 1024`; at
1.10 it moves to `1024 < M* <= 4096`. The small model fails at all three and at 2.0.

**Amendment: the endpoint.** The alignment endpoint in RESEARCH_PLAN.md section 5 is
superseded. Competence is defined by R. Algorithmic alignment requires a measure that
reads model behaviour rather than target loss, and no such measure has been validated,
so it is not being defined here by assertion.

**Phase 1 is paused** pending a bounded Phase 0B validation stage, pre-registered in
`PHASE_0B_PLAN.md`. Phase 0B trains nothing and needs no cluster time.

**Open items carried into Phase 0B and Phase 1 design.**

- Pool sampling and model initialization are currently driven by one seed and are
  confounded. At M=4 the two seeds give reference gaps of 30.08 and 40.96, a 36 percent
  swing from the pool draw alone. Split the RNGs, pair pool draws across model sizes,
  and define whether a seed success rate counts pool draws, initializations, or both.
- The training budget comparison rule is undeclared. All cells received 50,000 steps.
  Equal steps, equal examples, equal compute and training to convergence give different
  causal readings of H2 and one must be chosen.
- Evaluation logs only the mean over 8 batches of 512 prompts, so no uncertainty is
  available. The medium M=1024 seed 1 window clears the tolerance by 7e-5. Log
  per-prompt losses, estimate uncertainty by paired bootstrap at the independent task
  level, and require the confidence bound rather than the point estimate to satisfy the
  criterion.
- Ever-crossed endpoints are subject to selection across roughly a hundred overlapping
  candidate windows and should be treated as descriptive unless given simultaneous
  intervals or a held-out confirmation.
- Verification item: at M=4 seed 1 the small and medium models report unseen loss
  agreeing to within nine parts per million, 10.403086 against 10.403178, where every
  other paired cell differs by 5 to 135 percent. A four-task pool driving both models to
  the same in-weights solution is plausible, but the evaluation path should be checked.
- Every number in this entry was produced by ad hoc scripts against the raw JSONL.
  `analysis/plot_pilot.py` has never run. Reproduce all of it from a committed script
  before any of it is used in a manuscript.

---

## 2026-08-19: Project created

- Repository scaffolded: research plan pre-registered (RESEARCH_PLAN.md),
  library (tasks, model, closed-form ridge and dMMSE baselines, training
  loop), pilot config, SLURM array script for NMSU Discovery, smoke tests.
- Design decisions locked before any training: primary endpoint is the
  alignment score A on unseen tasks with emergence cutoff A <= 0.25;
  endpoint hierarchy frozen; H1 replication is the pilot gate.
- No training runs yet. Next step: run the smoke tests, then launch the
  Phase 0 pilot grid (28 cells: 2 model sizes x 7 pool sizes x 2 seeds).
