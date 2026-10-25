"""Performance and risk evaluation (standard, published methods).

* Annualised return/vol, Sharpe, Sortino, max drawdown, Calmar, turnover.
* Probabilistic and Deflated Sharpe Ratio (Bailey & Lopez de Prado 2012, 2014)
  to account for the number of strategy variants tried.
* OLS with Newey-West (1987) HAC t-stats for factor attribution.
* Kupiec (1995) proportion-of-failures test for VaR backtesting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


def drawdown(returns: pd.Series) -> pd.Series:
    wealth = (1 + returns.fillna(0)).cumprod()
    return wealth / wealth.cummax() - 1


def summary(returns: pd.Series, rf: pd.Series | None = None) -> dict:
    r = returns.dropna()
    ex = r - (rf.reindex(r.index).fillna(0) if rf is not None else 0)
    ann_ret = (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1
    vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = ex.mean() / ex.std() * np.sqrt(TRADING_DAYS)
    downside = ex[ex < 0].std() * np.sqrt(TRADING_DAYS)
    mdd = drawdown(r).min()
    return {
        "ann_return": ann_ret,
        "ann_vol": vol,
        "sharpe": sharpe,
        "sortino": ex.mean() * TRADING_DAYS / downside if downside else np.nan,
        "max_drawdown": mdd,
        "calmar": ann_ret / abs(mdd) if mdd else np.nan,
        "hit_rate": (r > 0).mean(),
        "skew": stats.skew(r),
        "excess_kurtosis": stats.kurtosis(r),
        "n_days": len(r),
    }


def probabilistic_sharpe(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true per-period SR > benchmark), adjusting for skew and kurtosis."""
    r = returns.dropna()
    sr = r.mean() / r.std()
    g3, g4 = stats.skew(r), stats.kurtosis(r, fisher=False)
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(len(r) - 1) / denom))


def deflated_sharpe(returns: pd.Series, n_trials: int, trial_sr_var: float | None = None) -> float:
    """DSR: PSR against the max SR expected from n_trials unskilled variants."""
    r = returns.dropna()
    if trial_sr_var is None:  # default: variance of SR estimator under null
        trial_sr_var = 1.0 / len(r)
    e = np.euler_gamma
    n = max(n_trials, 2)
    sr0 = np.sqrt(trial_sr_var) * ((1 - e) * stats.norm.ppf(1 - 1 / n) + e * stats.norm.ppf(1 - 1 / (n * np.e)))
    return probabilistic_sharpe(r, sr0)


def ols_newey_west(y: pd.Series, X: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """OLS with intercept; HAC (Bartlett kernel) standard errors."""
    Xi = np.column_stack([np.ones(len(X)), X.to_numpy()])
    yv = y.to_numpy()
    beta, *_ = np.linalg.lstsq(Xi, yv, rcond=None)
    u = yv - Xi @ beta
    T = len(yv)
    XtX_inv = np.linalg.inv(Xi.T @ Xi)
    Xu = Xi * u[:, None]
    S = Xu.T @ Xu
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        G = Xu[l:].T @ Xu[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    r2 = 1 - (u @ u) / ((yv - yv.mean()) @ (yv - yv.mean()))
    out = pd.DataFrame({"coef": beta, "t_stat": beta / se}, index=["alpha"] + list(X.columns))
    out.attrs["r2"] = r2
    out.attrs["n"] = T
    return out


def kupiec_pof(losses: pd.Series, var: pd.Series, alpha: float) -> dict:
    """Kupiec test: are VaR breaches as frequent as 1-alpha predicts?"""
    df = pd.concat([losses.rename("loss"), var.rename("var")], axis=1).dropna()
    n, x = len(df), int((df["loss"] > df["var"]).sum())
    p = 1 - alpha
    phat = x / n if n else np.nan
    if x in (0, n):
        lr = -2 * (n * np.log(1 - p) if x == 0 else n * np.log(p))
    else:
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - phat) - x * np.log(phat))
    return {"n": n, "breaches": x, "expected": n * p, "breach_rate": phat, "lr_stat": lr, "p_value": 1 - stats.chi2.cdf(lr, 1)}


def ic_summary(ics: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "mean_ic": ics.mean(),
        "ic_ir_ann": ics.mean() / ics.std() * np.sqrt(TRADING_DAYS),
        "t_stat": ics.mean() / ics.std() * np.sqrt(ics.count()),
        "pct_positive": (ics > 0).mean(),
    })
