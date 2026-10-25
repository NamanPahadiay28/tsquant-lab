"""Alpha signals ("models" in the SEC order's language).

The SEC order says Two Sigma runs "hundreds of different Models" that "create
forecasts". Their content is not disclosed. These are standard, published
cross-sectional equity signals used as stand-ins:

* momentum_12_1   Jegadeesh & Titman (1993): 12-month return skipping the last month.
* reversal_5d     Lehmann (1990), Jegadeesh (1990): short-term reversal.
* low_vol         Ang, Hodrick, Xing & Zhang (2006): low idiosyncratic/total volatility.
* low_beta        Frazzini & Pedersen (2014): betting against beta.
* resid_momentum  Blitz, Huij & Martens (2011): momentum in market-residual returns.

Every signal value at date t uses data up to and including t's close only.
Positions formed at t earn returns from t+1 (enforced in backtest.py).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

SignalFn = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]


def momentum_12_1(prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices.shift(21) / prices.shift(252))


def reversal_5d(prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    return -np.log(prices / prices.shift(5))


def low_vol(prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    return -prices.pct_change(fill_method=None).rolling(63, min_periods=42).std()


def _rolling_beta(rets: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    mkt = rets.mean(axis=1)
    cov = rets.rolling(window, min_periods=window // 2).cov(mkt)
    var = mkt.rolling(window, min_periods=window // 2).var()
    return cov.div(var, axis=0)


def low_beta(prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    return -_rolling_beta(prices.pct_change(fill_method=None))


def resid_momentum(prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    rets = prices.pct_change(fill_method=None)
    mkt = rets.mean(axis=1)
    beta = _rolling_beta(rets)
    resid = rets - beta.shift(1).mul(mkt, axis=0)
    cum = resid.rolling(231, min_periods=200).sum().shift(21)  # months t-12..t-2
    vol = resid.rolling(231, min_periods=200).std().shift(21)
    return cum / vol


SIGNALS: dict[str, SignalFn] = {
    "momentum_12_1": momentum_12_1,
    "reversal_5d": reversal_5d,
    "low_vol": low_vol,
    "low_beta": low_beta,
    "resid_momentum": resid_momentum,
}


def cross_sectional_zscore(raw: pd.DataFrame, winsor: float = 3.0) -> pd.DataFrame:
    """Demean and scale each date's cross-section, then clip outliers."""
    z = raw.sub(raw.mean(axis=1), axis=0).div(raw.std(axis=1), axis=0)
    z = z.clip(-winsor, winsor)
    return z.sub(z.mean(axis=1), axis=0).div(z.std(axis=1), axis=0)


def compute_signals(prices: pd.DataFrame, volume: pd.DataFrame, names: list[str]) -> dict[str, pd.DataFrame]:
    """Z-scored signal panels for the requested names."""
    out = {}
    for n in names:
        if n not in SIGNALS:
            raise KeyError(f"unknown signal {n!r}; available: {sorted(SIGNALS)}")
        out[n] = cross_sectional_zscore(SIGNALS[n](prices, volume))
    return out


def forward_returns(prices: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Return from t's close to t+horizon's close, aligned at t (for IC only)."""
    return prices.shift(-horizon) / prices - 1.0


def information_coefficient(signal: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Daily Spearman rank IC between a signal and forward returns."""
    s = signal.rank(axis=1)
    f = fwd.reindex_like(signal).rank(axis=1)
    mask = s.notna() & f.notna()
    s, f = s.where(mask), f.where(mask)
    s = s.sub(s.mean(axis=1), axis=0)
    f = f.sub(f.mean(axis=1), axis=0)
    ic = (s * f).sum(axis=1) / np.sqrt((s**2).sum(axis=1) * (f**2).sum(axis=1))
    return ic.where(mask.sum(axis=1) >= 10)
