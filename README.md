# tsquant-lab

A runnable, end-to-end systematic equity research pipeline built from what
Two Sigma's SEC filings actually disclose, with the gaps filled by standard,
published quant techniques.

> **Important.** Two Sigma has not filed its quant or risk models with the SEC.
> Its filings (Form ADV Part 2A brochures, 13F holdings, and the January 2025
> SEC settlement order) describe the *process* at a high level: many models
> that produce forecasts, a quantitative optimizer, position limits, VaR,
> stress tests, factor measurement, and model-approval governance. No formulas
> or parameters are public. Everything here that goes beyond those
> descriptions is labelled as a standard technique and cited. See
> [docs/FINDINGS.md](docs/FINDINGS.md).

## Pipeline

```
free data ──► signals (5 published anomalies) ──► governance gate (PAM-style correlation test, parameter lock)
          ──► decorrelation + IC-weighted combination ──► expected returns (Grinold)
          ──► Barra-style factor risk model ──► constrained mean-variance optimizer
          ──► costs (spread + square-root impact + borrow) ──► event-loop backtest
          ──► evaluation (Sharpe, DSR, VaR/CVaR + Kupiec, stress tests, macro factor lens attribution)
```

| Stage | Module | What the filings say | What we use |
|---|---|---|---|
| Data | `data.py`, `edgar.py` | Market + alternative data | Yahoo Finance, Ken French, FRED, SEC 13F |
| Models / forecasts | `signals.py` | "hundreds of different Models" create forecasts | Momentum, residual momentum, reversal, low vol, low beta |
| Governance | `governance.py` | PAM approval with a correlation threshold; unreviewed parameter edits caused the 2025 case | Correlation gate, hashed parameter lock, change log |
| Combination | `combine.py` | "decorrelation parameters"; strategies combined and weighted | Sequential residualisation, IC weights, Grinold alpha |
| Risk model | `risk.py` | VaR, stress tests, factor measurements | Barra-style factor model, PCA, Ledoit-Wolf, VaR/CVaR |
| Optimizer | `optimizer.py` | optimizer sets target positions; limits on leverage, turnover, positions | Closed-form constrained mean-variance, caps, vol target, partial trading |
| Costs | `costs.py` | frequent trading, high commissions, borrow costs, capacity | Half-spread, square-root impact, borrow fee |
| Evaluation | `evaluate.py`, `factor_lens.py` | Two Sigma Factor Lens, regression-based | Lens rebuilt from ETFs, Newey-West OLS, DSR, Kupiec |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[data,dev]"

pytest -q                                   # 14 tests incl. look-ahead checks
tsquant approve --reason "initial parameters"
tsquant run --synthetic                     # offline, ~25 s
tsquant run                                 # real data (downloads + caches)
```

Outputs land in `outputs/<synthetic|real>/`: `metrics.json`, `tearsheet.png`,
`daily.csv`, `weights.csv`, `ic_summary.csv`, `signal_corr.csv`,
`lens_exposures.csv`, `style_exposures.csv`, `stress.csv`, `admission.json`.

Other commands:

```bash
tsquant admit --candidate low_beta --synthetic        # would this signal pass the gate?
tsquant edgar-13f --user-agent "Your Name you@mail.com"   # Two Sigma's latest 13F top holdings
```

Changing any model parameter in `configs/default.yaml` makes `tsquant run`
refuse to start until you `tsquant approve --reason "..."`. That is the
control the SEC order says was missing. Use `--allow-unapproved` for
exploratory research only.

## Known limitations (read before trusting a number)

* **Survivorship bias**: the default universe is today's large caps. Build a
  point-in-time universe for real conclusions.
* **Yahoo data** is free but unofficial; adjustments can change.
* **No fundamentals**: value/quality signals need point-in-time accounting
  data (SEC XBRL "Financial Statement Data Sets" are a free route).
* **Synthetic results are a plumbing check**, not evidence of edge.
* **Multiple testing**: set `evaluation.n_trials` honestly; the Deflated
  Sharpe Ratio uses it.

See [docs/WORKFLOW.md](docs/WORKFLOW.md) for the step-by-step replication guide.
