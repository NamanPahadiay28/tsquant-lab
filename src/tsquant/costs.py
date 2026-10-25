"""Transaction cost and financing model (standard techniques, our choice).

The ADV Part 2A says the primary strategies "involve frequent trading" with
"significantly higher commissions" and that hedging costs such as the cost of
borrow can be prohibitive, and that profitability is expected to decrease as
assets grow (capacity). No cost model is disclosed.

We use:
* half-spread cost, linear in traded notional;
* square-root market impact, cost = k * sigma * sqrt(Q / ADV) per unit traded
  (Almgren, Thum, Hauptmann & Li 2005; Kyle & Obizhaeva 2016), which makes
  costs grow with AUM so capacity can be studied;
* borrow fee on short positions, charged daily.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def trading_cost(
    trades: pd.Series,
    nav: float,
    daily_vol: pd.Series,
    adv_dollars: pd.Series,
    half_spread_bps: float,
    impact_coef: float,
) -> float:
    """Cost of a rebalance as a fraction of NAV."""
    t = trades.abs()
    spread = t.sum() * half_spread_bps / 1e4
    participation = (t * nav / adv_dollars.reindex(t.index)).replace([np.inf], np.nan).fillna(1.0)
    impact = (t * impact_coef * daily_vol.reindex(t.index).fillna(daily_vol.median()) * np.sqrt(participation)).sum()
    return float(spread + impact)


def borrow_cost(weights: pd.Series, annual_bps: float) -> float:
    """Daily fee on the short book as a fraction of NAV."""
    return float(-weights.clip(upper=0).sum() * annual_bps / 1e4 / 252)
