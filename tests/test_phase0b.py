"""Phase 0B measure smoke tests. No training, no cluster, seconds to run."""

from __future__ import annotations

import math

import torch

from emerge.phase0b import build_probes, ridge_weights, score
from emerge.tasks import TaskSampler

DIM, SIGMA, TAU = 8, 0.25, 1.0


def _probe(pool_size=64, seed=0, family="pool", k=8, n=512):
    sampler = TaskSampler(DIM, SIGMA, pool_size, seed)
    g = torch.Generator().manual_seed(1234)
    return sampler, build_probes(sampler, n, k, family, SIGMA, g)


def test_references_score_at_their_own_ends():
    _, p = _probe()
    r = score(p["pred_ridge"], p["pred_ridge"], p["pred_dmmse"], TAU)
    d = score(p["pred_dmmse"], p["pred_ridge"], p["pred_dmmse"], TAU)
    assert abs(r["S"]) < 1e-9 and r["E"] < 1e-9
    assert abs(d["S"] - 1.0) < 1e-9 and d["E"] < 1e-9


def test_mixtures_recover_alpha_exactly():
    _, p = _probe()
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        mix = a * p["pred_dmmse"] + (1 - a) * p["pred_ridge"]
        s = score(mix, p["pred_ridge"], p["pred_dmmse"], TAU)
        assert abs(s["S"] - a) < 1e-9, (a, s["S"])
        assert s["E"] < 1e-9


def test_query_maximizes_divergence():
    """The chosen query beats a random query of the same norm, on average."""
    sampler, p = _probe()
    g = torch.Generator().manual_seed(7)
    rnd = torch.randn(p["x_q"].shape, generator=g)
    rnd = rnd / rnd.norm(dim=-1, keepdim=True) * math.sqrt(DIM)
    gap = p["w_dmmse"] - p["w_ridge"]
    chosen = (p["pred_dmmse"] - p["pred_ridge"]).abs().mean()
    random = (gap * rnd).sum(-1).abs().mean()
    assert chosen > random


def test_score_ignores_the_target_entirely():
    """S and E must not move when the targets change but predictions do not."""
    _, p = _probe()
    probe_pred = 0.3 * p["pred_dmmse"] + 0.7 * p["pred_ridge"]
    a = score(probe_pred, p["pred_ridge"], p["pred_dmmse"], TAU)
    p["y"] = p["y"] + 5.0  # targets shifted; predictions untouched
    b = score(probe_pred, p["pred_ridge"], p["pred_dmmse"], TAU)
    assert a == b


def test_off_axis_predictor_has_large_residual():
    """A predictor orthogonal to the reference axis must not read as aligned."""
    _, p = _probe()
    d = p["pred_dmmse"] - p["pred_ridge"]
    orth = p["pred_ridge"] + d.flip(0)  # same scale, uncorrelated with d
    s = score(orth, p["pred_ridge"], p["pred_dmmse"], TAU)
    assert s["E"] > 0.25


def test_unseen_family_builds():
    _, p = _probe(family="unseen")
    assert p["pred_ridge"].shape == p["pred_dmmse"].shape
