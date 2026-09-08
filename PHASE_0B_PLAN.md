# EMERGE Phase 0B Plan (Pre-Registered)

**Stage:** 0B, measurement validation
**Status:** Pre-registered design, written before any Phase 0B code is run.
**Date:** 2026-09-08
**Relationship to RESEARCH_PLAN.md:** RESEARCH_PLAN.md stays frozen. This document
adds a validation stage that did not exist in it, and supersedes the alignment
endpoint defined in its section 5. The reason is recorded in EXPERIMENT_LOG.md.

---

## 1. Why this stage exists

The Phase 0 pilot established that the pre-registered alignment score is an
algebraic rescaling of relative loss. With

    R = L_model / L_ridge
    G = (L_dmmse - L_ridge) / L_ridge

the pre-registered score satisfies, identically,

    A = (R - 1) / G

so A carries no information about the model beyond its loss. A fixed cutoff on A
therefore encodes an accuracy requirement that changes with pool size, and cannot
identify a comparable event across the diversity sweep.

Phase 0B builds and validates a second measure that reads the model's **behaviour**
rather than its loss, so that competence and algorithmic alignment become two
genuinely separate quantities. Competence stays defined by R.

Phase 1 is paused until this stage completes.

## 2. The proposed measure

For a probe prompt with context C and query point x_q, three predictions are
available without ever consulting the target:

- `y_ridge`, the ridge prediction under the continuous prior with the known noise level
- `y_dmmse`, the dMMSE prediction under the discrete pool of the run
- `y_model`, the model prediction

Write the reference divergence `d = y_dmmse - y_ridge`. A probe is admitted only
when `|d| >= tau`, which is what makes the measure well conditioned by construction
and is the property the alignment score lacked.

Over the admitted probe set, the **behavioural alignment score** is the least
squares projection of the model onto the segment joining the two references:

    S = sum( (y_model - y_ridge) * d ) / sum( d^2 )

and the **off-axis residual**, in units of the reference separation, is

    E = sqrt( sum( (y_model - y_ridge - S*d)^2 ) / sum( d^2 ) )

S near 0 means the model follows ridge, S near 1 means it follows dMMSE. E measures
how much of the model's behaviour lies off the line between the two references at
all. A reading of S is interpretable only when `E <= E_max`.

Projection is used rather than averaging per-probe ratios, because ratios remain
unstable near small denominators even after thresholding.

**The property that matters:** S and E are computed from predictions alone. No
target value enters either expression. They cannot be an algebraic transform of
target loss, which is the failure mode Phase 0 uncovered.

## 3. Probe construction

Two families, both generated in closed form from the pool and the context.

**Family 1, pool-concentrating.** Draw the context from a pool task `w_i`, so the
dMMSE posterior concentrates on `w_i`. Choose the query direction that maximizes the
divergence, which for a fixed query norm is

    x_q proportional to (w_i - w_ridge(C))

where `w_ridge(C)` is the ridge weight estimate from the context. This makes the two
references disagree as sharply as the geometry allows.

**Family 2, out-of-pool.** Draw the context from a task never in the pool, so dMMSE
must fall back on the pool while ridge tracks the true task. Same query rule.

Both families are needed. Family 1 asks whether a model reproduces pool-restricted
inference when the pool explains the context. Family 2 asks whether it does so even
when the pool does not.

## 4. Validation battery

The measure is validated against predictors whose behaviour is known by construction.
Testing only against perfect ridge and perfect dMMSE is insufficient, because almost
any normalized score separates its own endpoints.

| Predictor | Construction | Expected reading |
|---|---|---|
| Perfect ridge | closed form | S near 0, E near 0 |
| Perfect dMMSE | closed form | S near 1, E near 0 |
| Null | predicts 0 everywhere | E large, S must be reported as uninterpretable |
| Noisy ridge | ridge plus Gaussian output noise | S near 0, E grows with noise |
| Noisy dMMSE | dMMSE plus Gaussian output noise | S near 1, E grows with noise |
| Mixtures | `alpha*dmmse + (1-alpha)*ridge`, alpha in {0, 0.25, 0.5, 0.75, 1} | S recovers alpha, monotone, E near 0 |
| Right loss, wrong behaviour | ridge with a mis-set regularization, and OLS on the context | low target loss, S near 0, E measurably nonzero |
| Pool memorizer | nearest single pool task rather than the posterior mixture | S near 1 with detectable E, distinguishable from true dMMSE |
| Generalizer off-ridge | unregularized OLS at short context | S near 0 at long context, drifting at short context |

The whole battery runs at every pool size in the pilot grid, M in {4, 16, 64, 256,
1024, 4096}, and at both pilot seeds.

## 5. Pass criteria

The measure is adopted only if all five hold.

1. **Incompetence cannot look algorithmic.** The null predictor never yields S inside
   the ridge band together with a small E.
2. **Stable across diversity.** For the perfect and mixture predictors, S does not
   drift with M beyond a declared tolerance. This is the property A failed.
3. **Mixtures recover monotonically.** S tracks alpha monotonically across the mixture
   family, within a declared tolerance, at every M.
4. **Noise does not change the inferred mechanism.** S for noisy ridge and noisy dMMSE
   stays within its band as output noise rises to a declared level, with only E growing.
5. **Not a transform of target loss.** Demonstrated analytically, since no target
   appears in S or E, and empirically: across the battery the (R, S) pairs must span a
   region of the plane rather than tracing a curve. A deterministic relation between R
   and S is a failure.

## 6. Parameters to freeze before running

None of these are set yet. They are frozen in a single commit before the first run,
and any later change is an amendment in EXPERIMENT_LOG.md.

- `tau`, the minimum admitted reference divergence
- number of probes per cell, and the query norm
- context lengths k at which probes are evaluated
- `E_max`, the residual bound above which S is not interpreted
- tolerance bands for criteria 2, 3 and 4
- the ridge band and dMMSE band edges in S

## 7. Cost and scope

Phase 0B trains nothing. Every predictor in the battery is closed form or trivial, and
the probes are generated from the pools directly. It runs on CPU in minutes and needs
no cluster allocation. That is deliberate: the point is to find out whether the measure
works before spending a grid on it.

## 8. Known constraint: the pilot saved no weights

The Phase 0 runs used `save_final_checkpoint: false` and no checkpoints exist on disk.
The new measure therefore cannot be applied retroactively to the pilot models. Phase 0B
validates the measure against synthetic predictors only, and the first reading on a
trained model comes from Phase 1 or from a re-run.

Phase 1 must retain checkpoints at the evaluation steps it intends to analyze, or the
same problem recurs one stage later.

## 9. What happens after

If the measure passes, Phase 1 is designed with two endpoints: competence by R, and
behavioural alignment by S with its residual bound, each with the timepoint,
persistence and seed rules recorded in EXPERIMENT_LOG.md.

If the measure fails, EMERGE reframes honestly around `M*_competence`, H2 becomes a
performance-scaling hypothesis rather than a claim about mechanism, and the project
states plainly that it measures generalization performance and not the memorization to
algorithm transition.
