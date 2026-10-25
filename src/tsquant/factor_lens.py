"""Macro factor lens, replicating the published Two Sigma Factor Lens recipe
with free ETF proxies.

Source: Two Sigma, "Introducing the Two Sigma Factor Lens" (2018), a research
paper on twosigma.com, not an SEC filing. The SEC-filed ADV Part 2A of Two
Sigma Investor Solutions (Venn) confirms the lens exists and that its analyses
are "regression-based", but gives no construction details.

From the paper:
* Core macro: Equity, Interest Rates, Credit, Commodities.
* Secondary macro: Foreign Currency, Emerging Markets, Equity Short
  Volatility, Local Inflation.
* Credit and Commodities are residualised against Equity and Interest Rates;
  each credit index is residualised separately, then combined at equal risk.
* Secondary factors are residualised against all four core factors.
* The residual keeps the intercept ("intercept plus error").
* Rolling three-year window of daily returns, exponentially weighted.

Our choices (not in the paper): the ETF proxies, a one-year half-life, refits
every 21 days with betas applied out of sample, inverse-vol for "equal risk".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsquant.evaluate import ols_newey_west
from tsquant.risk import ewma_weights

CORE = ["equity", "rates", "credit", "commodities"]
SECONDARY = ["fx", "emerging", "short_vol", "inflation"]


def _ew_betas(y: np.ndarray, X: np.ndarray, halflife: float) -> np.ndarray | None:
    """Exponentially weighted OLS slopes (intercept fitted but not returned)."""
    ok = ~(np.isnan(y) | np.isnan(X).any(axis=1))
    y, X = y[ok], X[ok]
    if len(y) < 63:
        return None
    w = np.sqrt(ewma_weights(len(y), halflife))
    Xi = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(Xi * w[:, None], y * w, rcond=None)
    return coef[1:]


def residualize(y: pd.Series, X: pd.DataFrame, window: int, halflife: float, refit_every: int) -> pd.Series:
    """Out-of-sample rolling residual: y_t - X_t . beta(estimated through t-1)."""
    y, X = y.align(X, join="inner", axis=0)
    yv, Xv = y.to_numpy(), X.to_numpy()
    out = np.full(len(yv), np.nan)
    betas = None
    for i in range(len(yv)):
        if i >= window // 2 and (betas is None or i % refit_every == 0):
            lo = max(0, i - window)
            betas = _ew_betas(yv[lo:i], Xv[lo:i], halflife)
        if betas is not None:
            out[i] = yv[i] - Xv[i] @ betas
    return pd.Series(out, index=y.index)


def _raw(rets: pd.DataFrame, spec) -> pd.Series:
    """One ticker -> its return; [a, b] -> long a / short b spread."""
    if isinstance(spec, str):
        return rets[spec]
    if len(spec) == 2:
        return rets[spec[0]] - rets[spec[1]]
    raise ValueError(spec)


def build_lens(lens_prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    inst = cfg["instruments"]
    window, hl, refit = cfg["window_days"], cfg["halflife_days"], cfg["refit_every"]
    rets = lens_prices.pct_change(fill_method=None)
    f = pd.DataFrame(index=rets.index)
    f["equity"] = _raw(rets, inst["equity"])
    f["rates"] = _raw(rets, inst["rates"])
    base = f[["equity", "rates"]]

    credit_parts = [residualize(rets[t], base, window, hl, refit) for t in inst["credit"]]
    inv_vol = [1 / p.rolling(252, min_periods=63).std().shift(1) for p in credit_parts]
    wsum = sum(inv_vol)
    f["credit"] = sum(p * v / wsum for p, v in zip(credit_parts, inv_vol))
    f["commodities"] = residualize(_raw(rets, inst["commodities"]), base, window, hl, refit)

    core = f[CORE]
    for name in SECONDARY:
        if name in inst:
            f[name] = residualize(_raw(rets, inst[name]), core, window, hl, refit)
    return f.dropna(how="all")


def lens_exposures(portfolio_returns: pd.Series, lens: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """Regress a return stream on the lens (OLS, Newey-West t-stats)."""
    df = pd.concat([portfolio_returns.rename("y"), lens], axis=1, sort=True).dropna()
    return ols_newey_west(df["y"], df.drop(columns="y"), lags=lags)
