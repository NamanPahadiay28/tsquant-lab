"""Portfolio construction.

What the filings say: the ADV Part 2A says the optimizer turns combined
strategies into a "target goal position" per instrument, adjusting for risks
tied to size, liquidity and sector exposure, and that portfolio managers set
leverage, turnover, position limits and overall risk. No objective function is
disclosed.

What this module implements (standard technique, our choice): closed-form
mean-variance (Markowitz 1952) with linear equality constraints (dollar- and
factor-neutral), then position caps, a gross-leverage cap, an ex-ante vol
target, and partial trading toward the target to control turnover (in the
spirit of Garleanu & Pedersen 2013). It is deliberately transparent; swap in
cvxpy for richer constraints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def constrained_mean_variance(alpha: np.ndarray, cov: np.ndarray, A: np.ndarray | None, risk_aversion: float) -> np.ndarray:
    """argmax a'w - (g/2) w'Sw  s.t.  A w = 0.

    w = (1/g) [S^-1 a - S^-1 A' (A S^-1 A')^-1 A S^-1 a]
    """
    Si_a = np.linalg.solve(cov, alpha)
    if A is None or len(A) == 0:
        return Si_a / risk_aversion
    Si_At = np.linalg.solve(cov, A.T)
    lam = np.linalg.solve(A @ Si_At, A @ Si_a)
    return (Si_a - Si_At @ lam) / risk_aversion


def target_portfolio(
    alpha: pd.Series,
    cov: pd.DataFrame,
    neutral_exposures: pd.DataFrame | None,
    target_vol_daily: float,
    max_weight: float,
    max_gross: float,
    risk_aversion: float = 1.0,
    iterations: int = 5,
) -> pd.Series:
    names = alpha.dropna().index.intersection(cov.index)
    a = alpha[names].to_numpy()
    S = cov.loc[names, names].to_numpy()
    S = S + np.eye(len(names)) * 1e-10  # numerical ridge
    A = None
    if neutral_exposures is not None and neutral_exposures.shape[1]:
        A = neutral_exposures.reindex(names).fillna(0.0).to_numpy().T
    w = constrained_mean_variance(a, S, A, risk_aversion)
    for _ in range(iterations):
        vol = np.sqrt(w @ S @ w)
        if vol > 0:
            w *= target_vol_daily / vol
        w = np.clip(w, -max_weight, max_weight)
        if A is not None:  # project back onto the neutrality constraints after clipping
            w = w - A.T @ np.linalg.lstsq(A @ A.T, A @ w, rcond=None)[0]
        gross = np.abs(w).sum()
        if gross > max_gross:
            w *= max_gross / gross
    return pd.Series(w, index=names)


def partial_trade(current: pd.Series, target: pd.Series, trade_rate: float) -> pd.Series:
    """Move a fraction of the way from current to target holdings."""
    idx = current.index.union(target.index)
    c, t = current.reindex(idx).fillna(0.0), target.reindex(idx).fillna(0.0)
    return c + trade_rate * (t - c)
