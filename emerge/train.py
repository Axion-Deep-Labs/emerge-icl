"""Train one EMERGE cell: (model size, task pool size M, seed).

Usage:
  python -m emerge.train --config configs/pilot.yaml \
      --model-size small --pool-size 16 --seed 0

Writes results/<run_name>/metrics.jsonl with the model's eval loss and the
closed-form ridge and dMMSE baselines at every eval step, plus the primary
alignment score A defined in RESEARCH_PLAN.md section 5.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import yaml

from emerge.baselines import dmmse_predictions, mse_per_position, ridge_predictions
from emerge.model import ICLTransformer
from emerge.tasks import TaskSampler


def eval_steps(train_steps: int, interval: int) -> list[int]:
    """Log-spaced steps early (to resolve onset) plus a fixed interval."""
    steps = {0, train_steps}
    v = 1.0
    while v < train_steps:
        steps.add(int(round(v)))
        v *= 10 ** (1 / 12)
    steps.update(range(interval, train_steps, interval))
    return sorted(steps)


def lr_at(step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def model_losses(model, batches, device) -> torch.Tensor:
    """Mean per-position MSE (K,) over a list of fixed eval batches."""
    model.eval()
    acc = None
    for x, y_clean, y_noisy in batches:
        preds = model(x.to(device), y_noisy.to(device))
        per_pos = mse_per_position(preds, y_clean.to(device))
        acc = per_pos if acc is None else acc + per_pos
    model.train()
    return (acc / len(batches)).cpu()


def build_eval_set(sampler, cfg, seed, fresh_tasks):
    """Fixed eval batches (x, y_clean, y_noisy) plus baseline per-position MSEs.

    Baselines see the same noisy context the model sees and are scored, like
    the model, against the noise-free targets w.x.
    """
    g = torch.Generator().manual_seed(seed * 1_000_003 + (97 if fresh_tasks else 101))
    batches, ridge_acc, dmmse_acc = [], None, None
    for _ in range(cfg["eval_batches"]):
        x, y_noisy, w = sampler.sample(
            cfg["eval_batch_size"], cfg["n_points"], generator=g, fresh_tasks=fresh_tasks
        )
        y_clean = torch.einsum("bkd,bd->bk", x, w)
        batches.append((x, y_clean, y_noisy))
        r = mse_per_position(ridge_predictions(x, y_noisy, cfg["sigma"]), y_clean)
        ridge_acc = r if ridge_acc is None else ridge_acc + r
        if sampler.pool is not None:
            d = mse_per_position(
                dmmse_predictions(x, y_noisy, sampler.pool, cfg["sigma"]), y_clean
            )
            dmmse_acc = d if dmmse_acc is None else dmmse_acc + d
    n = len(batches)
    return (
        batches,
        ridge_acc / n,
        dmmse_acc / n if dmmse_acc is not None else None,
    )


def window_mean(per_pos: torch.Tensor, window: list[int]) -> float:
    return per_pos[window[0] : window[1]].mean().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-size", required=True)
    ap.add_argument("--pool-size", required=True, help="integer M, or 'inf'")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--steps", type=int, default=None, help="override train_steps")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.steps is not None:
        cfg["train_steps"] = args.steps
    size = cfg["model_sizes"][args.model_size]
    pool_size = None if args.pool_size == "inf" else int(args.pool_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    run_name = f"{args.model_size}_M{args.pool_size}_s{args.seed}"
    run_dir = Path(args.out) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    sampler = TaskSampler(cfg["dim"], cfg["sigma"], pool_size, args.seed)
    model = ICLTransformer(
        cfg["dim"], cfg["n_points"], size["d_model"], size["n_layers"], size["n_heads"]
    ).to(device)

    unseen, unseen_ridge, unseen_dmmse = build_eval_set(sampler, cfg, args.seed, True)
    if pool_size is not None:
        seen, seen_ridge, seen_dmmse = build_eval_set(sampler, cfg, args.seed, False)
    else:
        seen = None

    resolved = {**cfg, "model_size": args.model_size, "pool_size": args.pool_size,
                "seed": args.seed, "n_params": model.n_params(), "device": device}
    (run_dir / "config.json").write_text(json.dumps(resolved, indent=2))

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    evals = set(eval_steps(cfg["train_steps"], cfg["eval_interval"]))
    window = cfg["align_window"]
    running_loss = None

    with open(run_dir / "metrics.jsonl", "w") as fout:
        for step in range(cfg["train_steps"] + 1):
            if step in evals:
                lm_u = model_losses(model, unseen, device)
                rec = {
                    "step": step,
                    "train_loss": running_loss,
                    "unseen_model": window_mean(lm_u, window),
                    "unseen_ridge": window_mean(unseen_ridge, window),
                    "unseen_model_per_pos": [round(v, 6) for v in lm_u.tolist()],
                }
                if unseen_dmmse is not None:
                    lr_ = rec["unseen_ridge"]
                    ld = window_mean(unseen_dmmse, window)
                    rec["unseen_dmmse"] = ld
                    rec["alignment_A"] = (
                        (rec["unseen_model"] - lr_) / (ld - lr_)
                        if abs(ld - lr_) > 1e-9
                        else None
                    )
                if seen is not None:
                    lm_s = model_losses(model, seen, device)
                    rec["seen_model"] = window_mean(lm_s, window)
                    rec["seen_ridge"] = window_mean(seen_ridge, window)
                    rec["seen_dmmse"] = window_mean(seen_dmmse, window)
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
            if step == cfg["train_steps"]:
                break

            for group in opt.param_groups:
                group["lr"] = lr_at(step, cfg["lr"], cfg["warmup_steps"], cfg["train_steps"])
            x, y, _ = sampler.sample(cfg["batch_size"], cfg["n_points"])
            x, y = x.to(device), y.to(device)
            preds = model(x, y)
            loss = ((preds - y) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            running_loss = (
                loss.item()
                if running_loss is None
                else 0.99 * running_loss + 0.01 * loss.item()
            )

    if cfg.get("save_final_checkpoint", False):
        torch.save(model.state_dict(), run_dir / "final.pt")
    print(f"done: {run_name} ({model.n_params():,} params) -> {run_dir}")


if __name__ == "__main__":
    main()
