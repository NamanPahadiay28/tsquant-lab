"""Event-loop backtest with strict timing.

Timing convention: signals, risk model and target weights are formed with data
through the close of day t; the resulting holdings earn day t+1's return.
Costs are charged on the rebalance day. Between rebalances, weights drift with
prices. Returns are long/short P&L on NAV, excluding interest on cash.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tsquant.costs import borrow_cost, trading_cost
from tsquant.optimizer import partial_trade, target_portfolio
from tsquant.risk import StyleFactorModel, parametric_var


@dataclass
class BacktestResult:
    daily: pd.DataFrame  # gross, cost, borrow, net, turnover, exante_vol, exante_var99
    weights: pd.DataFrame  # holdings after each rebalance
    risk_snapshots: list[dict] = field(default_factory=list)


def run_backtest(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    alpha: pd.DataFrame,
    risk_model: StyleFactorModel,
    cfg: dict,
) -> BacktestResult:
    bt, opt, cc = cfg["backtest"], cfg["optimizer"], cfg["costs"]
    dates = returns.index
    start = dates.get_loc(alpha.dropna(how="all").index[0]) + bt.get("extra_warmup_days", 0)
    daily_vol = returns.rolling(63, min_periods=21).std()
    adv = (prices * volume).rolling(21, min_periods=5).mean()
    nav = float(bt["nav_usd"])
    target_vol_daily = opt["target_vol_annual"] / np.sqrt(252)

    w = pd.Series(0.0, index=returns.columns)
    cov = None
    rows, weights, snaps = [], {}, []
    for i in range(start, len(dates) - 1):
        t, t1 = dates[i], dates[i + 1]
        cost = 0.0
        turnover = 0.0
        if (i - start) % bt["rebalance_every"] == 0:
            rm = risk_model.at(t)
            a = alpha.loc[t].dropna()
            if rm is not None and len(a) > 10:
                cov = rm.covariance()
                neutral = rm.exposures[opt["neutralize"]] if opt["neutralize"] else None
                target = target_portfolio(
                    a, cov, neutral, target_vol_daily, opt["max_weight"], opt["max_gross"], opt["risk_aversion"]
                )
                new_w = partial_trade(w, target, opt["trade_rate"]).reindex(w.index).fillna(0.0)
                trades = new_w - w
                turnover = float(trades.abs().sum())
                cost = trading_cost(trades, nav, daily_vol.loc[t], adv.loc[t], cc["half_spread_bps"], cc["impact_coef"])
                w = new_w
                weights[t] = w
                if bt.get("snapshot_every_rebalances") and len(weights) % bt["snapshot_every_rebalances"] == 0:
                    snaps.append({"date": t, **rm.decompose(w)})
        r1 = returns.loc[t1].fillna(0.0)
        gross = float((w * r1).sum())
        borrow = borrow_cost(w, cc["borrow_bps_annual"])
        ex_vol = float(np.sqrt(w @ cov.reindex(index=w.index, columns=w.index).fillna(0) @ w)) if cov is not None else np.nan
        ex_var = parametric_var(w, cov, 0.99) if cov is not None else np.nan
        rows.append({"date": t1, "gross": gross, "cost": cost, "borrow": borrow, "net": gross - cost - borrow,
                     "turnover": turnover, "exante_vol": ex_vol, "exante_var99": ex_var})
        # Let weights drift with prices until the next rebalance.
        w = w * (1 + r1) / (1 + gross) if gross > -1 else w
    daily = pd.DataFrame(rows).set_index("date")
    traded = daily["turnover"] > 0
    if traded.any():  # drop the warm-up stretch before the first trade
        daily = daily.loc[traded.idxmax():]
    return BacktestResult(daily, pd.DataFrame(weights).T, snaps)
