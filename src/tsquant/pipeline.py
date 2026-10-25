"""End-to-end pipeline: data -> signals -> governance -> combine -> risk ->
optimize -> backtest -> evaluate. Each stage maps to a section of
docs/WORKFLOW.md."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from tsquant import combine as cmb
from tsquant import evaluate as ev
from tsquant import risk
from tsquant.backtest import run_backtest
from tsquant.data import MarketData
from tsquant.factor_lens import build_lens, lens_exposures
from tsquant.governance import admission_test
from tsquant.signals import _rolling_beta, compute_signals, cross_sectional_zscore, forward_returns, information_coefficient

log = logging.getLogger(__name__)


def run_pipeline(md: MarketData, cfg: dict) -> dict:
    sc, cc, rc = cfg["signals"], cfg["combination"], cfg["risk"]
    returns = md.returns

    log.info("signals: %s", sc["names"])
    signals = compute_signals(md.prices, md.volume, sc["names"])

    # Governance: PAM-style admission test, in listed order, against the
    # signals already admitted. Rejected signals are dropped if enforced.
    g = cfg["governance"]
    live, admission = [sc["names"][0]], []
    for n in sc["names"][1:]:
        a = admission_test(n, signals, live, g["max_abs_corr"])
        admission.append(a)
        if a["admitted"] or not g.get("enforce_admission", True):
            live.append(n)
        else:
            log.info("admission gate rejected %s (max |corr| %.2f)", n, a["max_abs_corr"])
    corr = pd.DataFrame(
        {a: {b: _corr(signals[a], signals[b]) for b in sc["names"]} for a in sc["names"]}
    )

    log.info("decorrelate + combine: %s", live)
    dec = cmb.decorrelate({n: signals[n] for n in live}, live, cc["decorrelation_strength"])
    weights = cmb.ic_weights(dec, md.prices, cc["ic_horizon"], cc["ic_lookback"], cc["shrink_to_equal"])
    composite = cmb.combine(dec, weights)
    alpha = cmb.expected_returns(composite, md.prices, cc["target_ic"])

    log.info("risk model")
    beta_z = cross_sectional_zscore(_rolling_beta(returns))
    exposures = {"beta": beta_z, **{n: signals[n] for n in rc["style_factors"]}}
    model = risk.StyleFactorModel(returns, exposures, rc["halflife_factor"], rc["halflife_specific"], rc["window"])

    log.info("backtest")
    bt = run_backtest(returns, md.prices, md.volume, alpha, model, cfg)
    net, gross = bt.daily["net"], bt.daily["gross"]

    log.info("evaluate")
    fwd = forward_returns(md.prices, 1)
    ics = pd.DataFrame({n: information_coefficient(s, fwd) for n, s in signals.items()})
    ics["composite"] = information_coefficient(composite, fwd)
    lens = build_lens(md.lens_prices, cfg["factor_lens"])
    lens_beta = lens_exposures(net, lens)
    style_beta = ev.ols_newey_west(*_align(net, model.factor_returns.drop(columns="market")))

    var_h, cvar_h = risk.var_cvar(net, 0.99, "historical")
    var_cf, cvar_cf = risk.var_cvar(net, 0.99, "cornish_fisher")
    var_g, cvar_g = risk.var_cvar(net, 0.99, "gaussian")
    kupiec = ev.kupiec_pof(-net, bt.daily["exante_var99"], 0.99)

    last_w = bt.weights.iloc[-1]
    stress_hist = risk.historical_stress(last_w, returns, {k: tuple(v) for k, v in cfg["stress"]["historical"].items()})
    last_rm = model.at(bt.weights.index[-1])
    decomposition = last_rm.decompose(last_w)
    stress_fac = risk.factor_shock(pd.Series(decomposition["exposures"]), cfg["stress"]["factor_shocks"])

    n_trials = cfg["evaluation"]["n_trials"]
    metrics = {
        "net": ev.summary(net),
        "gross": ev.summary(gross),
        "avg_daily_turnover": float(bt.daily["turnover"].sum() / len(bt.daily)),
        "annual_cost_drag": float((bt.daily["cost"] + bt.daily["borrow"]).mean() * 252),
        "psr_vs_0": ev.probabilistic_sharpe(net),
        "deflated_sharpe": ev.deflated_sharpe(net, n_trials),
        "n_trials_assumed": n_trials,
        "live_signals": live,
        "var99": {"historical": [var_h, cvar_h], "gaussian": [var_g, cvar_g], "cornish_fisher": [var_cf, cvar_cf]},
        "kupiec_exante_var99": kupiec,
        "lens_r2": lens_beta.attrs["r2"],
        "style_r2": style_beta.attrs["r2"],
        "last_risk_decomposition": decomposition,
    }
    return {
        "metrics": metrics,
        "daily": bt.daily,
        "weights": bt.weights,
        "ic_summary": ev.ic_summary(ics),
        "signal_corr": corr,
        "admission": admission,
        "combo_weights": weights,
        "lens": lens,
        "lens_exposures": lens_beta,
        "style_exposures": style_beta,
        "stress_historical": stress_hist,
        "stress_factor": stress_fac,
        "factor_returns": model.factor_returns,
    }


def _corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
    from tsquant.governance import mean_signal_correlation

    return mean_signal_correlation(a, b)


def _align(y: pd.Series, X: pd.DataFrame):
    df = pd.concat([y.rename("y"), X], axis=1, sort=True).dropna()
    return df["y"], df.drop(columns="y")
