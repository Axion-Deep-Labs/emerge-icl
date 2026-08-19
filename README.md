# EMERGE

**Empirical Mapping of Emergent Regimes in GEneralization**

When does in-context learning emerge in transformers trained from scratch, and how does the emergence threshold scale with task diversity and model capacity?

An open research program by [Axion Deep Labs](https://axiondeep.com). Everything here is MIT licensed: code, configs, analysis, and released results. All data is synthetic and generated on the fly from seeded RNGs. No external datasets, no pretrained weights.

## The question

Train a small transformer from scratch on noisy linear regression prompts, where each prompt is a sequence of (x, y) examples from one task. If pretraining only ever shows the model M distinct tasks, does it memorize the pool or learn the general algorithm?

This setting has exact optimal baselines, which is why we chose it:

- A pool memorizer should match **dMMSE**, the Bayes predictor over the discrete pool.
- A true in-context learner should match **ridge regression**, the Bayes predictor over the continuous task prior.

Our alignment score A places every trained model on the line between those two references. The program maps how the crossover point M* (the task diversity where in-context learning wins) scales with model capacity and training compute, and whether the transition is sharp or gradual in training time.

The full pre-registered design, hypotheses, endpoints, and analysis plan are in [RESEARCH_PLAN.md](RESEARCH_PLAN.md). The plan was committed before the first training run.

## Quickstart

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m emerge.train --config configs/pilot.yaml --model-size small --pool-size 16 --seed 0
```

A single pilot cell runs in minutes on a consumer GPU and produces `results/<run_name>/metrics.jsonl` with model loss and both closed-form baselines at every eval step.

Smoke test (CPU, under a minute):

```bash
.venv/bin/python -m pytest tests/ -q
```

## Repository layout

```
RESEARCH_PLAN.md      Pre-registered design (frozen; amendments logged)
EXPERIMENT_LOG.md     Dated log of every run batch and decision
configs/              Experiment configs (YAML)
emerge/               Library: tasks, model, baselines, training, eval
scripts/slurm/        SLURM scripts for the NMSU Discovery cluster
analysis/             Figure and statistics scripts (regenerate every figure)
results/              Run outputs (gitignored; released as artifacts with the paper)
tests/                Smoke tests
```

## Compute

Phase 0 pilot runs on a single consumer GPU. Phase 1 runs as SLURM arrays on an NVIDIA A100 at the NMSU Discovery HPC cluster. Every run is reproducible from its config and seed.

## Status

Phase 0 (pilot) is in progress. See EXPERIMENT_LOG.md for the current state.

## License

MIT. See [LICENSE](LICENSE).
