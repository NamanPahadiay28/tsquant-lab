"""Signal decorrelation and combination into one expected-return forecast.

What the filings say: the SEC order says modelers used "decorrelation
parameters" to reduce a model's correlation with other models, because
redundant models could make the firm "buy or sell more or less of a security
than intended". The ADV Part 2A says strategies are combined into
"quantitatively-optimized portfolios" with weights set by modelers, research
teams and portfolio managers. Neither discloses formulas.

What this module does (standard techniques, our choice):
* Sequential cross-sectional residualisation with a strength ``lambda`` in
  [0, 1] per signal: 0 = use the raw signal, 1 = fully orthogonal to the
  signals ranked above it. This is our reading of a "decorrelation parameter".
* Weights from trailing information coefficients (IC), shrunk toward equal
  weight, using only ICs whose forward window has already closed.
* Grinold's rule, alpha = IC x volatility x score (Grinold & Kahn, 2000), to
  turn a z-score into an expected return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsquant.signals import forward_returns, information_coefficient


def decorrelate(signals: dict[str, pd.DataFrame], order: list[str], strength: dict[str, float]) -> dict[str, pd.DataFrame]:
    """Residualise each signal on those before it in ``order``, date by date."""
    names = [n for n in order if n in signals]
    idx, cols = signals[names[0]].index, signals[names[0]].columns
    stack = np.stack([signals[n].reindex(index=idx, columns=cols).to_numpy() for n in names], axis=2)
    out = stack.copy()
    for t in range(stack.shape[0]):
        X = stack[t]
        ok = ~np.isnan(X).any(axis=1)
        if ok.sum() < len(names) + 5:
            continue
        for k in range(1, len(names)):
            lam = strength.get(names[k], 0.0)
            if lam == 0.0:
                continue
            A = out[t, ok, :k]
            y = X[ok, k]
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            out[t, ok, k] = y - lam * (A @ beta)
    res = {}
    for k, n in enumerate(names):
        df = pd.DataFrame(out[:, :, k], index=idx, columns=cols)
        res[n] = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)
    return res


def ic_weights(
    signals: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    horizon: int,
    lookback: int,
    shrink: float,
    allow_negative: bool = False,
) -> pd.DataFrame:
    """Per-date combination weights from trailing ICs (no look-ahead)."""
    fwd = forward_returns(prices, horizon)
    ics = pd.DataFrame({n: information_coefficient(s, fwd) for n, s in signals.items()})
    # IC at t needs prices at t+horizon, so it is only known after t+horizon.
    known = ics.shift(horizon + 1)
    mean_ic = known.rolling(lookback, min_periods=lookback // 2).mean()
    if not allow_negative:
        mean_ic = mean_ic.clip(lower=0.0)
    w_ic = mean_ic.div(mean_ic.abs().sum(axis=1).replace(0, np.nan), axis=0)
    equal = 1.0 / len(signals)
    w = (1 - shrink) * w_ic.fillna(equal) + shrink * equal
    return w.div(w.abs().sum(axis=1), axis=0)


def combine(signals: dict[str, pd.DataFrame], weights: pd.DataFrame) -> pd.DataFrame:
    """Weighted sum of signal z-scores, re-standardised cross-sectionally."""
    total = None
    for n, s in signals.items():
        term = s.fillna(0.0).mul(weights[n], axis=0)
        total = term if total is None else total + term
    z = total.sub(total.mean(axis=1), axis=0).div(total.std(axis=1), axis=0)
    return z


def expected_returns(composite_z: pd.DataFrame, prices: pd.DataFrame, ic: float, vol_window: int = 63) -> pd.DataFrame:
    """Grinold's rule: alpha_i = IC * sigma_i * z_i (per-period expected return)."""
    vol = prices.pct_change(fill_method=None).rolling(vol_window, min_periods=vol_window // 2).std()
    return ic * vol * composite_z
