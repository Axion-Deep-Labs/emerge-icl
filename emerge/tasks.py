"""Synthetic noisy linear regression tasks.

A task is a weight vector w ~ N(0, I_d). A prompt is K (x, y) pairs from one
task with x ~ N(0, I_d) and y = w.x + eps, eps ~ N(0, sigma^2). All data is
generated on the fly from seeded generators; there is no external dataset.
"""

from __future__ import annotations

import torch


class TaskSampler:
    """Samples prompt batches from a finite task pool or the continuous prior.

    pool_size None means the continuous prior: every prompt gets a fresh task.
    """

    def __init__(self, dim: int, sigma: float, pool_size: int | None, seed: int):
        self.dim = dim
        self.sigma = sigma
        self.pool_size = pool_size
        # The pool is a property of the run seed and is drawn once, so every
        # batch and every evaluation of this run sees the same M tasks.
        pool_gen = torch.Generator().manual_seed(seed * 1_000_003 + 17)
        self.pool = (
            torch.randn(pool_size, dim, generator=pool_gen)
            if pool_size is not None
            else None
        )
        self.gen = torch.Generator().manual_seed(seed * 1_000_003 + 29)

    def _draw_tasks(self, batch: int, generator: torch.Generator) -> torch.Tensor:
        if self.pool is None:
            return torch.randn(batch, self.dim, generator=generator)
        idx = torch.randint(self.pool_size, (batch,), generator=generator)
        return self.pool[idx]

    def sample(
        self,
        batch: int,
        n_points: int,
        generator: torch.Generator | None = None,
        fresh_tasks: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns x (B, K, d), y (B, K), w (B, d).

        fresh_tasks=True draws tasks from the continuous prior regardless of
        the pool. That is the unseen-task evaluation distribution.
        """
        g = generator if generator is not None else self.gen
        if fresh_tasks:
            w = torch.randn(batch, self.dim, generator=g)
        else:
            w = self._draw_tasks(batch, g)
        x = torch.randn(batch, n_points, self.dim, generator=g)
        noise = torch.randn(batch, n_points, generator=g) * self.sigma
        y = torch.einsum("bkd,bd->bk", x, w) + noise
        return x, y, w
