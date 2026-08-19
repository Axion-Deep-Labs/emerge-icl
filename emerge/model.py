"""Decoder-only transformer for in-context regression, trained from scratch.

Prompt encoding follows the standard in-context regression setup: the
sequence x_1, y_1, ..., x_K, y_K is embedded as 2K tokens in R^(d+1),
where an x token is [x, 0] and a y token is [0, ..., 0, y]. The model
predicts a scalar at every position; predictions at x positions are the
in-context estimates of y and carry the loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b, t, c = h.shape
        q, k, v = self.qkv(self.ln1(h)).chunk(3, dim=-1)
        q, k, v = (
            z.view(b, t, self.n_heads, c // self.n_heads).transpose(1, 2)
            for z in (q, k, v)
        )
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h + self.proj(a.transpose(1, 2).reshape(b, t, c))
        h = h + self.mlp(self.ln2(h))
        return h


class ICLTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        n_points: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
    ):
        super().__init__()
        self.dim = dim
        self.read_in = nn.Linear(dim + 1, d_model)
        self.pos = nn.Embedding(2 * n_points, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.read_out = nn.Linear(d_model, 1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x (B, K, d), y (B, K) -> predictions at x positions, (B, K)."""
        b, k, d = x.shape
        tokens = x.new_zeros(b, 2 * k, d + 1)
        tokens[:, 0::2, :d] = x
        tokens[:, 1::2, d] = y
        h = self.read_in(tokens) + self.pos(
            torch.arange(2 * k, device=x.device)
        )
        for block in self.blocks:
            h = block(h)
        out = self.read_out(self.ln_f(h)).squeeze(-1)
        return out[:, 0::2]
