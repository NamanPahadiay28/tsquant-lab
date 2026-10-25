import numpy as np
import pandas as pd
import pytest

from tsquant import combine as cmb
from tsquant import edgar, evaluate, risk
from tsquant.backtest import run_backtest
from tsquant.data import MarketData, _parse_french_csv
from tsquant.factor_lens import build_lens
from tsquant.governance import ParamRegistry, UnapprovedParametersError, admission_test, model_params
from tsquant.optimizer import target_portfolio
from tsquant.pipeline import run_pipeline
from tsquant.signals import SIGNALS, compute_signals, cross_sectional_zscore, _rolling_beta


def _perturb_after(md: MarketData, t: pd.Timestamp, seed: int = 0) -> MarketData:
    """Replace every price after t with a different random path."""
    rng = np.random.default_rng(seed)
    p = md.prices.copy()
    after = p.index > t
    shocks = np.exp(np.cumsum(rng.normal(0, 0.03, (after.sum(), p.shape[1])), axis=0))
    p.loc[after] = p.loc[t].to_numpy() * shocks
    return MarketData(p, md.volume, md.lens_prices, md.rf_daily)


def test_signals_have_no_lookahead(md):
    t = md.prices.index[600]
    alt = _perturb_after(md, t)
    for name, fn in SIGNALS.items():
        a = fn(md.prices, md.volume).loc[:t]
        b = fn(alt.prices, alt.volume).loc[:t]
        pd.testing.assert_frame_equal(a, b, obj=name)


def test_combined_alpha_has_no_lookahead(md, cfg):
    t = md.prices.index[700]
    alt = _perturb_after(md, t)

    def alpha(m):
        s = compute_signals(m.prices, m.volume, ["momentum_12_1", "reversal_5d", "low_vol"])
        d = cmb.decorrelate(s, list(s), {"low_vol": 1.0})
        w = cmb.ic_weights(d, m.prices, 5, 126, 0.5)
        return cmb.expected_returns(cmb.combine(d, w), m.prices, 0.03).loc[:t]

    pd.testing.assert_frame_equal(alpha(md), alpha(alt))


def test_backtest_pnl_has_no_lookahead(md, cfg):
    """P&L through t must not change when prices after t change."""
    t = md.prices.index[750]
    alt = _perturb_after(md, t)

    def pnl(m):
        sig = compute_signals(m.prices, m.volume, ["momentum_12_1", "reversal_5d"])
        w = cmb.ic_weights(sig, m.prices, 5, 126, 0.5)
        a = cmb.expected_returns(cmb.combine(sig, w), m.prices, 0.03)
        expo = {"beta": cross_sectional_zscore(_rolling_beta(m.returns)), **sig}
        model = risk.StyleFactorModel(m.returns, expo, 90, 60, 252)
        return run_backtest(m.returns, m.prices, m.volume, a, model, cfg).daily.loc[:t]

    pd.testing.assert_frame_equal(pnl(md), pnl(alt))


def test_decorrelate_full_strength_removes_correlation(md):
    s = compute_signals(md.prices, md.volume, ["low_vol", "low_beta"])
    d = cmb.decorrelate(s, ["low_vol", "low_beta"], {"low_beta": 1.0})
    a, b = d["low_vol"].iloc[400:], d["low_beta"].iloc[400:]
    c = a.T.corrwith(b.T).dropna()
    assert c.abs().max() < 1e-8


def test_admission_rejects_near_duplicate(md):
    s = compute_signals(md.prices, md.volume, ["momentum_12_1", "resid_momentum", "reversal_5d"])
    assert not admission_test("resid_momentum", s, ["momentum_12_1"], 0.5)["admitted"]
    assert admission_test("reversal_5d", s, ["momentum_12_1"], 0.5)["admitted"]


def test_param_registry_blocks_unapproved_changes(cfg):
    reg = ParamRegistry(cfg["governance"]["lock_file"], cfg["governance"]["change_log"])
    params = model_params(cfg)
    with pytest.raises(UnapprovedParametersError):
        reg.verify(params)
    reg.approve(params, "initial", "tester")
    reg.verify(params)
    cfg["combination"]["decorrelation_strength"]["low_beta"] = 0.0  # the SEC-order scenario
    with pytest.raises(UnapprovedParametersError, match="combination.decorrelation_strength.low_beta"):
        reg.verify(model_params(cfg))
    reg.approve(model_params(cfg), "reviewed", "tester")
    log = open(cfg["governance"]["change_log"]).read().strip().splitlines()
    assert len(log) == 2 and "low_beta" in log[1]


def test_optimizer_respects_constraints():
    rng = np.random.default_rng(1)
    n = 50
    names = [f"S{i}" for i in range(n)]
    B = rng.normal(size=(n, 3))
    cov = pd.DataFrame(B @ B.T * 1e-4 + np.eye(n) * 2e-4, index=names, columns=names)
    alpha = pd.Series(rng.normal(0, 1e-3, n), index=names)
    neutral = pd.DataFrame({"market": 1.0, "beta": rng.normal(1, 0.3, n)}, index=names)
    w = target_portfolio(alpha, cov, neutral, 0.01, 0.05, 3.0)
    assert abs(w.sum()) < 1e-10
    assert abs(w @ neutral["beta"]) < 1e-10
    assert np.abs(w).sum() <= 3.0 + 1e-9
    assert np.abs(w).max() <= 0.05 * 1.25  # re-projection may nudge slightly past the cap


def test_var_cvar_gaussian_sample():
    x = np.random.default_rng(0).normal(0, 0.01, 200_000)
    var_h, es_h = risk.var_cvar(x, 0.99, "historical")
    var_g, es_g = risk.var_cvar(x, 0.99, "gaussian")
    assert var_h == pytest.approx(0.02326, rel=0.02)
    assert var_g == pytest.approx(0.02326, rel=0.02)
    assert es_h == pytest.approx(0.02665, rel=0.03)


def test_covariance_models_are_psd(md):
    r = md.returns.iloc[-300:]
    lw, delta = risk.ledoit_wolf(r)
    assert 0 <= delta <= 1
    assert np.linalg.eigvalsh(lw.to_numpy()).min() > 0
    pca = risk.pca_factor_model(r, 5).covariance()
    assert np.linalg.eigvalsh(pca.to_numpy()).min() > 0


def test_lens_residuals_are_orthogonal_to_core(md, cfg):
    lens = build_lens(md.lens_prices, cfg["factor_lens"]).dropna()
    for f in ["credit", "commodities"]:
        for base in ["equity", "rates"]:
            assert abs(lens[f].corr(lens[base])) < 0.15, (f, base)
    for f in ["fx", "emerging", "inflation", "short_vol"]:
        assert lens[["equity", "rates", "credit", "commodities"]].corrwith(lens[f]).abs().max() < 0.15


def test_kupiec_and_dsr_sanity():
    rng = np.random.default_rng(2)
    loss = pd.Series(rng.normal(0, 1, 5000))
    k = evaluate.kupiec_pof(loss, pd.Series(2.326, index=loss.index), 0.99)
    assert k["p_value"] > 0.01
    noise = pd.Series(rng.normal(0, 0.01, 2000))
    assert evaluate.deflated_sharpe(noise, 100) < 0.5


def test_parse_13f_information_table():
    xml = """<?xml version="1.0"?>
    <informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
      <infoTable><nameOfIssuer>APPLE INC</nameOfIssuer><titleOfClass>COM</titleOfClass>
        <cusip>037833100</cusip><value>1000</value>
        <shrsOrPrnAmt><sshPrnamt>10</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt>
        <investmentDiscretion>SOLE</investmentDiscretion></infoTable>
      <infoTable><nameOfIssuer>APPLE INC</nameOfIssuer><titleOfClass>CALL</titleOfClass>
        <cusip>037833100</cusip><value>500</value>
        <shrsOrPrnAmt><sshPrnamt>5</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt>
        <putCall>Call</putCall><investmentDiscretion>SOLE</investmentDiscretion></infoTable>
      <infoTable><nameOfIssuer>MICROSOFT CORP</nameOfIssuer><titleOfClass>COM</titleOfClass>
        <cusip>594918104</cusip><value>3000</value>
        <shrsOrPrnAmt><sshPrnamt>7</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt>
        <investmentDiscretion>SOLE</investmentDiscretion></infoTable>
    </informationTable>"""
    h = edgar.parse_information_table(xml)
    assert len(h) == 3
    top = edgar.top_holdings(h, 10)
    assert list(top["nameOfIssuer"]) == ["MICROSOFT CORP", "APPLE INC"]
    assert top["weight"].sum() == pytest.approx(1.0)


def test_parse_french_csv():
    text = "This file was created\n\n,Mkt-RF,SMB,HML,RF\n19260701,    0.09,   -0.25,   -0.27,    0.01\n19260702,    0.45,   -0.33,   -0.06,    0.01\n\nCopyright"
    df = _parse_french_csv(text)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert df.iloc[0]["Mkt-RF"] == pytest.approx(0.0009)


def test_end_to_end_smoke(md, cfg):
    res = run_pipeline(md, cfg)
    m = res["metrics"]
    assert np.isfinite(m["net"]["sharpe"])
    assert "resid_momentum" not in m["live_signals"]  # rejected by the admission gate
    assert res["daily"]["net"].abs().max() < 0.2
