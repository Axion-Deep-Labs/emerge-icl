# EMERGE Experiment Log

Dated log of every run batch, decision, and amendment to the pre-registered
plan. Newest entries at the top.

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
