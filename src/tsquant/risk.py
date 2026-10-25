"""Risk layer: covariance models, VaR/CVaR, stress tests, risk decomposition.

What the filings say: Two Sigma's ADV Part 2A says risk is controlled mainly
through "strategy weightings, position limits", controls built into each
optimizer, and monitoring with "Value at Risk (VaR) and similar calculations,
stress-testing", model correlation metrics and factor measurements, with a
Chief Risk Officer as an independent check. No risk-model formulas are filed.

What this module implements (standard, published techniques):
* Fundamental-style factor model (Barra approach; Grinold & Kahn 2000):
  daily cross-sectional regressions of returns on yesterday's exposures give
  factor returns; Sigma = X F X' + D with EWMA estimates of F and D.
* Statistical (PCA) factor model (Connor & Korajczyk 1986).
* Ledoit-Wolf (2004) shrinkage covariance as a benchmark estimator.
* VaR / CVaR: historical, Gaussian, and Cornish-Fisher (modified) VaR.
* Historical-scenario and hypothetical factor-shock stress tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class FactorRisk:
    exposures: pd.DataFrame  # n x k
    factor_cov: pd.DataFrame  # k x k
    specific_var: pd.Series  # n

    def covariance(self) -> pd.DataFrame:
        X = self.exposures.to_numpy()
        S = X @ self.factor_cov.to_numpy() @ X.T + np.diag(self.specific_var.to_numpy())
        return pd.DataFrame(S, index=self.exposures.index, columns=self.exposures.index)

    def decompose(self, w: pd.Series) -> dict:
        """Split portfolio variance into factor and specific parts, with
        each factor's contribution (x_k * (F x)_k) where x = X'w."""
        w = w.reindex(self.exposures.index).fillna(0.0)
        x = self.exposures.T @ w
        Fx = self.factor_cov @ x
        factor_var = float(x @ Fx)
        spec_var = float((w**2 * self.specific_var).sum())
        total = factor_var + spec_var
        return {
            "total_vol": np.sqrt(total),
            "factor_share": factor_var / total if total else np.nan,
            "specific_share": spec_var / total if total else np.nan,
            "factor_contrib": (x * Fx / total).to_dict() if total else {},
            "exposures": x.to_dict(),
        }


def ewma_weights(n: int, halflife: float) -> np.ndarray:
    w = 0.5 ** (np.arange(n)[::-1] / halflife)
    return w / w.sum()


def ewma_cov(x: np.ndarray, halflife: float) -> np.ndarray:
    w = ewma_weights(len(x), halflife)
    mu = w @ x
    xc = x - mu
    return (xc * w[:, None]).T @ xc


class StyleFactorModel:
    """Barra-style model. Factors: market (intercept), beta, and style signals.

    ``fit_factor_returns`` runs one cross-sectional regression per day,
    r_t = X_{t-1} f_t + u_t, so factor returns at t use only exposures known
    at t-1. ``at(date)`` then builds the covariance from history up to date.
    """

    def __init__(self, returns: pd.DataFrame, exposures: dict[str, pd.DataFrame], halflife_factor: float, halflife_specific: float, window: int):
        self.returns = returns
        self.names = ["market"] + list(exposures)
        self.exposures = {k: v.reindex_like(returns) for k, v in exposures.items()}
        self.hl_f, self.hl_s, self.window = halflife_factor, halflife_specific, window
        self.factor_returns, self.residuals = self._fit()

    def _X(self, date) -> pd.DataFrame:
        cols = {"market": pd.Series(1.0, index=self.returns.columns)}
        for k, v in self.exposures.items():
            cols[k] = v.loc[date]
        return pd.DataFrame(cols)

    def _fit(self):
        dates = self.returns.index
        f = np.full((len(dates), len(self.names)), np.nan)
        u = np.full(self.returns.shape, np.nan)
        R = self.returns.to_numpy()
        expo = np.stack([np.ones(self.returns.shape)] + [e.to_numpy() for e in self.exposures.values()], axis=2)
        for t in range(1, len(dates)):
            X, r = expo[t - 1], R[t]
            ok = ~(np.isnan(X).any(axis=1) | np.isnan(r))
            if ok.sum() < len(self.names) + 10:
                continue
            coef, *_ = np.linalg.lstsq(X[ok], r[ok], rcond=None)
            f[t] = coef
            u[t, ok] = r[ok] - X[ok] @ coef
        return (
            pd.DataFrame(f, index=dates, columns=self.names),
            pd.DataFrame(u, index=dates, columns=self.returns.columns),
        )

    def at(self, date) -> FactorRisk | None:
        fr = self.factor_returns.loc[:date].dropna().iloc[-self.window :]
        if len(fr) < 60:
            return None
        F = pd.DataFrame(ewma_cov(fr.to_numpy(), self.hl_f), index=self.names, columns=self.names)
        res = self.residuals.loc[:date].iloc[-self.window :]
        w = ewma_weights(len(res), self.hl_s)
        spec = pd.Series(np.nansum((res.to_numpy() ** 2) * w[:, None], axis=0) / np.maximum(
            np.sum(~np.isnan(res.to_numpy()) * w[:, None], axis=0), 1e-12), index=res.columns)
        # Shrink specific variances toward the cross-sectional median (Bayesian-style).
        spec = 0.7 * spec + 0.3 * spec.median()
        X = self._X(date).fillna(0.0)
        return FactorRisk(X, F, spec.reindex(X.index).fillna(spec.median()))


def pca_factor_model(returns: pd.DataFrame, n_factors: int) -> FactorRisk:
    """Statistical factor model from the top principal components."""
    r = returns.dropna(axis=1, how="any")
    r = r - r.mean()
    U, s, Vt = np.linalg.svd(r.to_numpy(), full_matrices=False)
    B = Vt[:n_factors].T  # n x k loadings
    f = r.to_numpy() @ B  # T x k factor returns
    resid = r.to_numpy() - f @ B.T
    names = [f"pc{i+1}" for i in range(n_factors)]
    return FactorRisk(
        pd.DataFrame(B, index=r.columns, columns=names),
        pd.DataFrame(np.cov(f, rowvar=False).reshape(n_factors, n_factors), index=names, columns=names),
        pd.Series(resid.var(axis=0, ddof=1), index=r.columns),
    )


def ledoit_wolf(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Ledoit & Wolf (2004) shrinkage toward a scaled identity. Returns (cov, delta)."""
    X = returns.dropna().to_numpy()
    T, n = X.shape
    X = X - X.mean(axis=0)
    S = X.T @ X / T
    mu = np.trace(S) / n
    d2 = np.linalg.norm(S - mu * np.eye(n), "fro") ** 2
    b2 = sum(np.linalg.norm(np.outer(x, x) - S, "fro") ** 2 for x in X) / T**2
    delta = min(b2, d2) / d2 if d2 > 0 else 1.0
    cov = delta * mu * np.eye(n) + (1 - delta) * S
    cols = returns.columns
    return pd.DataFrame(cov, index=cols, columns=cols), float(delta)


# ---------------------------------------------------------------- VaR / CVaR

def var_cvar(pnl: pd.Series | np.ndarray, alpha: float = 0.99, method: str = "historical") -> tuple[float, float]:
    """One-period VaR and CVaR as positive loss numbers.

    methods: 'historical', 'gaussian', 'cornish_fisher' (Favre & Galeano 2002).
    """
    x = pd.Series(pnl).dropna().to_numpy()
    if method == "historical":
        q = np.quantile(x, 1 - alpha)
        return -q, -x[x <= q].mean()
    mu, sd = x.mean(), x.std(ddof=1)
    z = stats.norm.ppf(1 - alpha)
    if method == "gaussian":
        es = mu - sd * stats.norm.pdf(z) / (1 - alpha)
        return -(mu + z * sd), -es
    if method == "cornish_fisher":
        s, k = stats.skew(x), stats.kurtosis(x)  # excess kurtosis
        zcf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
        var = -(mu + zcf * sd)
        tail = x[x <= -var]
        return var, (-tail.mean() if len(tail) else var)
    raise ValueError(method)


def parametric_var(w: pd.Series, cov: pd.DataFrame, alpha: float = 0.99) -> float:
    """Ex-ante Gaussian VaR of a weight vector under a covariance matrix."""
    w = w.reindex(cov.index).fillna(0.0)
    return float(-stats.norm.ppf(1 - alpha) * np.sqrt(w @ cov @ w))


# ---------------------------------------------------------------- stress tests

def historical_stress(w: pd.Series, returns: pd.DataFrame, scenarios: dict[str, tuple[str, str]]) -> pd.Series:
    """Replay each historical window's cumulative asset returns on today's weights."""
    out = {}
    for name, (start, end) in scenarios.items():
        window = returns.loc[start:end]
        if window.empty:
            out[name] = np.nan
            continue
        cum = (1 + window.fillna(0.0)).prod() - 1
        out[name] = float((w.reindex(cum.index).fillna(0.0) * cum).sum())
    return pd.Series(out, name="pnl")


def factor_shock(exposures: pd.Series, shocks: dict[str, dict[str, float]]) -> pd.Series:
    """Hypothetical scenarios: P&L = sum_k exposure_k * shock_k."""
    return pd.Series({n: float(sum(exposures.get(k, 0.0) * v for k, v in s.items())) for n, s in shocks.items()}, name="pnl")
