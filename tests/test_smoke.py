"""CPU smoke tests: shapes, baseline sanity, and a tiny end-to-end run."""

import json
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emerge.baselines import dmmse_predictions, mse_per_position, ridge_predictions
from emerge.model import ICLTransformer
from emerge.tasks import TaskSampler

REPO = Path(__file__).resolve().parents[1]


def test_sampler_pool_is_fixed():
    s1 = TaskSampler(dim=4, sigma=0.1, pool_size=8, seed=3)
    s2 = TaskSampler(dim=4, sigma=0.1, pool_size=8, seed=3)
    assert torch.equal(s1.pool, s2.pool)
    x, y, w = s1.sample(16, 10)
    assert x.shape == (16, 10, 4) and y.shape == (16, 10) and w.shape == (16, 4)


def test_ridge_beats_zero_predictor():
    s = TaskSampler(dim=4, sigma=0.1, pool_size=None, seed=0)
    x, y_noisy, w = s.sample(256, 12)
    y_clean = torch.einsum("bkd,bd->bk", x, w)
    ridge = mse_per_position(ridge_predictions(x, y_noisy, 0.1), y_clean)
    zero = mse_per_position(torch.zeros_like(y_clean), y_clean)
    # With context, ridge must be far better than predicting zero.
    assert ridge[-1] < 0.5 * zero[-1]


def test_dmmse_recovers_pool_task():
    s = TaskSampler(dim=4, sigma=0.1, pool_size=8, seed=1)
    x, y_noisy, w = s.sample(256, 12)
    y_clean = torch.einsum("bkd,bd->bk", x, w)
    dmmse = mse_per_position(dmmse_predictions(x, y_noisy, s.pool, 0.1), y_clean)
    # On pool tasks with enough context, dMMSE identifies the task almost
    # exactly, so late-position error approaches zero.
    assert dmmse[-1] < 0.05


def test_model_forward_shape():
    m = ICLTransformer(dim=4, n_points=10, d_model=32, n_layers=2, n_heads=2)
    x, y = torch.randn(6, 10, 4), torch.randn(6, 10)
    assert m(x, y).shape == (6, 10)


def test_train_end_to_end(tmp_path):
    out = subprocess.run(
        [
            sys.executable, "-m", "emerge.train",
            "--config", "configs/pilot.yaml",
            "--model-size", "small",
            "--pool-size", "16",
            "--seed", "0",
            "--steps", "30",
            "--out", str(tmp_path),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    metrics = (tmp_path / "small_M16_s0" / "metrics.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in metrics]
    assert records[-1]["step"] == 30
    assert "alignment_A" in records[-1]
    assert records[-1]["unseen_ridge"] < records[-1]["unseen_dmmse"]
