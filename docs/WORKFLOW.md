# Replication workflow

Eight stages. For each: what to do, which file implements it, what the
filings support (**[Filing]**) and what is a standard technique we chose
(**[Standard]**). Read [FINDINGS.md](FINDINGS.md) first.

## 0. Setup

```bash
pip install -e ".[data,dev]"
pytest -q
tsquant approve --reason "initial parameters"
tsquant run --synthetic   # verify plumbing offline
tsquant run               # real data
```

## 1. Data sourcing — `data.py`, `edgar.py`

* Equity prices/volume: Yahoo Finance via `yfinance` (adjusted close).
* Macro lens ETFs: ACWI, IEF, LQD, HYG, DBC, UDN, EEM, TIP, PUTW.
* Risk-free: FRED `DTB3`. Benchmark factors: Ken French daily FF5 + momentum.
* SEC 13F holdings via EDGAR (`tsquant edgar-13f`).
* Everything is cached in `data/cache/` so reruns are reproducible.

[Filing] TSI uses market and alternative data. [Standard] all specific
sources. Upgrade path: point-in-time universe (historical index membership or
13F-based), SEC XBRL fundamentals for value/quality, CRSP if you have
university access.

## 2. Signals ("models") — `signals.py`

Five published cross-sectional signals, each z-scored per day and winsorised:
momentum 12-1, residual momentum, 5-day reversal, low volatility, low beta.
All use data through day t only; `tests/` checks this by scrambling future
prices.

[Filing] many models produce forecasts. [Standard] the signals themselves.

## 3. Governance — `governance.py`

* **Admission gate**: a new signal is admitted only if its average
  cross-sectional correlation with already-admitted signals is below
  `governance.max_abs_corr` (0.5). On synthetic data residual momentum (0.95
  correlated with momentum) is rejected.
* **Parameter lock**: model parameters are hashed; `tsquant run` refuses to
  run if they differ from the approved lock. `tsquant approve` writes the lock
  and appends who/when/why/which keys changed to `configs/param_changes.jsonl`.

[Filing] PAM / Mini-PAM with a correlation threshold, and the celFS failure
that let parameters change without review. [Standard] the hashing and log.

## 4. Decorrelation and combination — `combine.py`

1. Residualise each signal on the ones above it, scaled by a per-signal
   `decorrelation_strength` in [0, 1].
2. Weight signals by trailing 252-day mean IC (5-day horizon), lagged so only
   completed forward windows are used, shrunk 50% toward equal weight.
3. Expected return = IC × volatility × z-score (Grinold & Kahn).

[Filing] "decorrelation parameters"; strategy weights. [Standard] everything
numeric. Experiment: set a strength to 0 and watch the composite's exposure to
that signal rise — the 2025 case in miniature.

## 5. Risk model — `risk.py`

* Barra-style factor model: each day, regress returns on yesterday's
  exposures (market, beta, momentum, low-vol, reversal) to get factor returns
  and residuals; Σ = X F Xᵀ + D with EWMA F (half-life 90d) and D (60d, shrunk
  toward the median).
* Alternatives for comparison: PCA factor model, Ledoit-Wolf shrinkage.
* VaR/CVaR at 99%: historical, Gaussian, Cornish-Fisher; ex-ante parametric
  VaR from the model each day.
* Stress tests: replay historical windows (2007 quant quake, 2008, 2009
  momentum crash, COVID 2020, 2022 rates) on current weights, plus
  hypothetical factor shocks.
* Risk decomposition: factor vs specific share, per-factor contribution.

[Filing] VaR, stress testing, factor measurements, position limits.
[Standard] all models and settings.

## 6. Portfolio construction and costs — `optimizer.py`, `costs.py`

* Maximise αᵀw − ½γ wᵀΣw subject to dollar- and beta-neutrality (closed
  form), then cap positions at 5%, cap gross at 3×, scale to 10% annual
  ex-ante vol, and trade 50% of the way to target each rebalance (weekly).
* Costs: 2 bps half-spread, square-root impact (coef 0.1 × daily vol ×
  √(trade/ADV)) at $100M NAV, 50 bps/yr borrow on shorts.

[Filing] optimizer → target positions; PMs set leverage, turnover and
position limits; costs and capacity are risks. [Standard] objective,
constraints, cost model.

## 7. Backtest — `backtest.py`

Decide at t's close, earn t+1's return, charge costs on trade day, let
weights drift between rebalances. Output: daily gross/net P&L, turnover,
ex-ante vol and VaR.

## 8. Evaluation — `evaluate.py`, `factor_lens.py`, `report.py`

* Sharpe, Sortino, max drawdown, Calmar, hit rate, skew/kurtosis, turnover,
  cost drag.
* Probabilistic and Deflated Sharpe Ratio (set `n_trials` honestly).
* Signal IC table and signal correlation matrix.
* Kupiec test: does the ex-ante 99% VaR get breached ~1% of days?
* Attribution: regress net returns on (a) the rebuilt Two Sigma-style macro
  factor lens and (b) the style factors, with Newey-West t-stats. A good
  market-neutral book shows low lens R² and a positive alpha.

## Suggested extensions (each is a good portfolio project on its own)

1. Point-in-time universe from historical 13F filings or index membership.
2. Fundamentals from SEC XBRL → value, quality, accruals signals.
3. ML combination layer (ridge / gradient boosting on signals) with
   purged, embargoed cross-validation (López de Prado 2018).
4. Swap the closed-form optimizer for `cvxpy` with sector and turnover constraints.
5. Capacity curve: rerun at NAV $10M → $5B and plot net Sharpe.
6. Crowding: correlate your weights with changes in 13F holdings.
