"""Synthetic market generator so the whole pipeline runs offline and in CI.

The data has a known structure, which makes it a test bed: a market factor and
a few latent sector factors, slowly varying per-stock drifts (so momentum has a
small real edge), mild short-term mean reversion, and a low-volatility premium.
Macro ETF series for the factor lens are built from latent macro shocks with
known loadings, so the lens residualisation can be checked against the truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsquant.data import MarketData

LENS_TICKERS = ["ACWI", "IEF", "LQD", "HYG", "DBC", "UDN", "EEM", "TIP", "PUTW"]


def make_synthetic_market(
    n_stocks: int = 80, n_days: int = 2520, seed: int = 7, start: str = "2012-01-02"
) -> MarketData:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    tickers = [f"S{i:03d}" for i in range(n_stocks)]

    # Latent macro shocks (daily): equity, rates, credit, commodity, fx, em, vol, inflation.
    macro_vol = np.array([0.010, 0.004, 0.002, 0.009, 0.004, 0.006, 0.003, 0.002])
    macro = rng.standard_normal((n_days, 8)) * macro_vol
    macro[:, 0] += 0.0003  # equity premium
    eq, rates = macro[:, 0], macro[:, 1]

    # Stocks: market beta + 4 sector factors + idiosyncratic noise.
    beta = rng.uniform(0.5, 1.5, n_stocks)
    sector = rng.integers(0, 4, n_stocks)
    sector_f = rng.standard_normal((n_days, 4)) * 0.006
    idio_vol = rng.uniform(0.008, 0.025, n_stocks)
    eps = rng.standard_normal((n_days, n_stocks)) * idio_vol

    # Persistent drift (momentum edge): AR(1) with ~1y half-life.
    phi = 0.5 ** (1 / 252)
    drift = np.zeros((n_days, n_stocks))
    d = rng.standard_normal(n_stocks) * 0.0004
    for t in range(n_days):
        d = phi * d + np.sqrt(1 - phi**2) * rng.standard_normal(n_stocks) * 0.0004
        drift[t] = d

    # Low-vol premium: lower idio vol -> slightly higher drift.
    lowvol_premium = 0.00015 * (idio_vol.mean() - idio_vol) / idio_vol.std()

    rets = beta * eq[:, None] + sector_f[:, sector] + eps + drift + lowvol_premium
    # Short-term reversal: give back 8% of the previous day's idiosyncratic move.
    rets[1:] -= 0.08 * eps[:-1]
    prices = pd.DataFrame(100 * np.exp(np.cumsum(np.log1p(rets), axis=0)), index=dates, columns=tickers)
    adv = rng.lognormal(mean=15, sigma=1.0, size=n_stocks)  # shares/day
    volume = pd.DataFrame(adv * rng.lognormal(0, 0.3, (n_days, n_stocks)), index=dates, columns=tickers)

    # Lens ETFs: known loadings on the latent macro shocks.
    L = {
        "ACWI": [1.0, 0, 0, 0, 0, 0, 0, 0],
        "IEF": [0, 1.0, 0, 0, 0, 0, 0, 0],
        "LQD": [0.1, 0.8, 1.0, 0, 0, 0, 0, 0],
        "HYG": [0.4, 0.2, 2.0, 0, 0, 0, 0, 0],
        "DBC": [0.3, -0.1, 0, 1.0, 0, 0, 0, 0],
        "UDN": [0.1, 0, 0, 0, 1.0, 0, 0, 0],
        "EEM": [1.1, 0, 0, 0.1, 0.2, 1.0, 0, 0],
        "TIP": [0, 0.9, 0, 0, 0, 0, 0, 1.0],
        "PUTW": [0.5, 0, 0, 0, 0, 0, 1.0, 0],
    }
    noise = rng.standard_normal((n_days, len(L))) * 0.0005
    lens_rets = np.column_stack([macro @ np.array(v) for v in L.values()]) + noise
    lens_prices = pd.DataFrame(
        100 * np.exp(np.cumsum(np.log1p(lens_rets), axis=0)), index=dates, columns=list(L)
    )
    rf = pd.Series(0.02 / 252, index=dates)
    return MarketData(prices, volume, lens_prices, rf)
