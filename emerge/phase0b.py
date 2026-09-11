"""Phase 0B: a behavioural alignment measure, and the battery that validates it.

See PHASE_0B_PLAN.md. The Phase 0 alignment score turned out to satisfy
A = (R - 1) / G exactly, so it carried no information beyond relative loss.
The measure here reads model predictions against the two closed-form references
and never touches the target, so it cannot be a transform of target loss.

For a probe with context C and query x_q, write
    d = y_dmmse - y_ridge          the reference divergence
    m = y_model - y_ridge          the model's displacement from ridge
over the admitted probes. A probe is admitted when the two references disagree,
|d| >= tau, and when the ridge reference is itself distinguishable from silence,
|y_ridge| >= rho. The second condition was added on 2026-09-10; without it, a
short context shrinks ridge toward zero, predicting nothing reads as following
ridge, and the measure cannot tell competence from silence. Then
    S = sum(m * d) / sum(d * d)    least squares projection onto the reference axis
    E = sqrt(sum((m - S*d)^2) / sum(d*d))    off-axis residual, in units of d
S near 0 means the model follows ridge, S near 1 means it follows dMMSE, and S is
interpretable only when E is small: a model that follows neither reference has a
large residual and must not be read as aligned with either.
"""

from __future__ import annotations

import math

import torch

from .baselines import dmmse_predictions, ridge_predictions


@torch.no_grad()
def ridge_weights(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Posterior-mean weights under w ~ N(0, I) from the whole context. (B, d)."""
    _, _, d = x.shape
    eye = torch.eye(d, dtype=x.dtype, device=x.device)
    gram = x.transpose(1, 2) @ x + (sigma**2) * eye
    rhs = torch.einsum("bkd,bk->bd", x, y)
    return torch.linalg.solve(gram, rhs)


@torch.no_grad()
def ols_weights(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Minimum-norm least squares weights from the whole context. (B, d).

    The pseudoinverse rather than a solver, because short contexts leave the
    system underdetermined (context length 4 against 8 dimensions) and that
    regime is exactly where this predictor is supposed to differ from ridge.
    """
    return (torch.linalg.pinv(x) @ y.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def pool_posterior(
    x: torch.Tensor, y: torch.Tensor, pool: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Posterior over pool tasks given the whole context. (B, M)."""
    pool = pool.to(x.device, x.dtype)
    f = torch.einsum("bkd,md->bkm", x, pool)
    loglik = -((y.unsqueeze(-1) - f) ** 2).sum(dim=1) / (2 * sigma**2)
    return torch.softmax(loglik, dim=-1)


@torch.no_grad()
def dmmse_weights(
    x: torch.Tensor, y: torch.Tensor, pool: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Posterior-weighted mean pool task. (B, d).

    dMMSE is linear in the query, so its prediction at x_q is exactly this
    vector dotted with x_q. That is what makes the divergence-maximizing query
    direction available in closed form.
    """
    post = pool_posterior(x, y, pool, sigma)
    return post @ pool.to(x.device, x.dtype)


@torch.no_grad()
def build_probes(
    sampler,
    n_probes: int,
    context_len: int,
    family: str,
    sigma: float,
    generator: torch.Generator,
):
    """Build counterfactual probes where the two references disagree sharply.

    family "pool":   context drawn from a pool task, so the posterior concentrates
                     on it and dMMSE speaks with confidence.
    family "unseen": context drawn from the continuous prior, so dMMSE has to fall
                     back on a pool that does not contain the task.

    Both references are linear in the query, so the divergence at x_q is exactly
    (w_dmmse - w_ridge) . x_q and is maximized, at fixed query norm, by placing
    x_q along that difference.

    Returns a dict with the context, the query, and both reference predictions.
    """
    if sampler.pool is None:
        raise ValueError("probes need a task pool; M = infinity has no dMMSE")
    x, y, w_true = sampler.sample(
        n_probes, context_len, generator=generator, fresh_tasks=(family == "unseen")
    )
    w_ridge = ridge_weights(x, y, sigma)
    w_dmmse = dmmse_weights(x, y, sampler.pool, sigma)

    direction = w_dmmse - w_ridge
    norm = direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x_q = direction / norm * math.sqrt(sampler.dim)

    return {
        "x": x,
        "y": y,
        "x_q": x_q,
        "w_true": w_true,
        "w_ridge": w_ridge,
        "w_dmmse": w_dmmse,
        "pred_ridge": (w_ridge * x_q).sum(-1),
        "pred_dmmse": (w_dmmse * x_q).sum(-1),
        # Noiseless target at the query. Used only for the (R, S) independence
        # check; it plays no part in S or E.
        "target_q": (w_true * x_q).sum(-1),
    }


@torch.no_grad()
def score(
    pred_model: torch.Tensor,
    pred_ridge: torch.Tensor,
    pred_dmmse: torch.Tensor,
    tau: float,
    rho: float = 0.0,
) -> dict:
    """Behavioural alignment S and off-axis residual E over admitted probes.

    A probe is admitted when the references disagree, |d| >= tau, and when the
    ridge reference is non-degenerate, |y_ridge| >= rho. rho defaults to 0, which
    is the pre-2026-09-10 behaviour and is what the algebraic unit tests use.
    """
    d = pred_dmmse - pred_ridge
    m = pred_model - pred_ridge
    keep = (d.abs() >= tau) & (pred_ridge.abs() >= rho)
    n_kept = int(keep.sum())
    if n_kept == 0:
        return {"S": float("nan"), "E": float("nan"), "n_admitted": 0, "admit_rate": 0.0}
    d, m = d[keep].double(), m[keep].double()
    dd = (d * d).sum()
    s = (m * d).sum() / dd
    resid = m - s * d
    e = torch.sqrt((resid * resid).sum() / dd)
    return {
        "S": s.item(),
        "E": e.item(),
        "n_admitted": n_kept,
        "admit_rate": n_kept / keep.numel(),
    }


# --- The validation battery -------------------------------------------------
# Every entry maps a probe bundle to a prediction at the query. None of them is
# the model under study; they are predictors whose behaviour is known by
# construction, used to find out whether the measure reports behaviour correctly.


@torch.no_grad()
def battery_predictions(probe: dict, sampler, sigma: float, cfg: dict, gen: torch.Generator) -> dict:
    """name -> prediction at the query (B,)."""
    x, y, x_q = probe["x"], probe["y"], probe["x_q"]
    p_ridge, p_dmmse = probe["pred_ridge"], probe["pred_dmmse"]
    out: dict[str, torch.Tensor] = {
        "perfect_ridge": p_ridge,
        "perfect_dmmse": p_dmmse,
        "null": torch.zeros_like(p_ridge),
    }

    # Output noise on each reference: behaviour unchanged, accuracy degraded.
    for s in cfg["noise_levels"]:
        out[f"noisy_ridge_{s}"] = p_ridge + s * sigma * torch.randn(
            p_ridge.shape, generator=gen, dtype=p_ridge.dtype
        )
        out[f"noisy_dmmse_{s}"] = p_dmmse + s * sigma * torch.randn(
            p_dmmse.shape, generator=gen, dtype=p_dmmse.dtype
        )

    # Convex mixtures: S must recover alpha.
    for a in cfg["mixture_alphas"]:
        out[f"mix_{a}"] = a * p_dmmse + (1.0 - a) * p_ridge

    # Right loss, wrong behaviour: ridge with a mis-set regularizer, and OLS.
    c = cfg["wrong_lambda_multiple"]
    w_wrong = ridge_weights(x, y, sigma * c)
    out["wrong_lambda"] = (w_wrong * x_q).sum(-1)
    w_ols = ols_weights(x, y)
    out["ols"] = (w_ols * x_q).sum(-1)

    # Pool memorizer: the single most likely pool task, not the posterior mixture.
    post = pool_posterior(x, y, sampler.pool, sigma)
    w_nn = sampler.pool.to(x.device, x.dtype)[post.argmax(dim=-1)]
    out["pool_nearest"] = (w_nn * x_q).sum(-1)
    return out


@torch.no_grad()
def relative_loss(probe: dict, pred: torch.Tensor) -> float:
    """R = mean squared error of pred over that of ridge, at the query.

    Measured against the noiseless target so the noise floor does not inflate
    both terms and compress the ratio. R exists here only to demonstrate that S
    is not a function of it; neither S nor E consults any target.
    """
    t = probe["target_q"].double()
    num = ((pred.double() - t) ** 2).mean()
    den = ((probe["pred_ridge"].double() - t) ** 2).mean()
    return (num / den).item()
