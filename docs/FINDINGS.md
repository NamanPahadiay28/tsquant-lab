# What Two Sigma's SEC filings disclose

Research date: 2026-10-25. Each item separates
**what the document says** from **our inference**.

## Bottom line

Two Sigma has **not** "submitted quant risk models to the SEC". No public
filing contains model code, formulas, factor definitions or parameters.
Advisers describe their methods in Form ADV Part 2A at a high level, and
the firm's models came up publicly mainly through a **2025 SEC enforcement
order** about weak controls over model parameters. That order is the most
detailed public description of how Two Sigma's models are organised, but it
is about governance, not about how the models forecast returns.

## Sources reviewed

| # | Document | Filed by | Link |
|---|---|---|---|
| 1 | SEC order, Admin. Proc. File No. 3-22418; Exchange Act Rel. 102207; Advisers Act Rel. 6824 (Jan 16, 2025) | SEC | https://www.sec.gov/files/litigation/admin/2025/34-102207.pdf |
| 2 | SEC press release 2025-15 | SEC | https://www.sec.gov/newsroom/press-releases/2025-15 |
| 3 | Form ADV Part 2A brochure, Two Sigma Investments, LP (Mar 31, 2025) | TSI | https://files.adviserinfo.sec.gov/IAPD/Content/Common/crd_iapd_Brochure.aspx?BRCHR_VRSN_ID=971935 |
| 4 | Form ADV Part 2A brochure, Two Sigma Investor Solutions, LP (Venn) (Mar 31, 2025) | TSIS | https://app.venn.twosigma.com/api/legal/doc/adv |
| 5 | Form 13F-HR filings, Two Sigma Investments (CIK 1179392) | TSI | e.g. https://www.sec.gov/Archives/edgar/data/1179392/0000919574-23-001463-index.htm |
| 6 | "Introducing the Two Sigma Factor Lens" (2018) — **research paper, not an SEC filing** | Two Sigma | https://www.twosigma.com/wp-content/uploads/Introducing-the-Two-Sigma-Factor-Lens.10.18.pdf |

## 1. SEC order (January 2025)

**What it says**
* Two Sigma Investments (TSI) and Two Sigma Advisers (TSA) use "hundreds of
  different Models" built by researchers ("modelers") that "create forecasts"
  used for investment decisions. TSI licenses the models to TSA.
* Model code lives in a controlled file ("the Jar"). Because machine-learning
  models needed more data, some parameters were stored in a database called
  **celFS**; these parameters could change how strongly a model affected trading.
* Modelers used **"decorrelation parameters"** to reduce a model's correlation
  with other models, removing redundancy that could otherwise make the firm
  trade more or less of a security than intended.
* New models went through **PAM** ("Productionalize a Model"): a white paper
  plus forms summarising the model and its **correlation to existing models**;
  models could be approved if that correlation was below a set threshold.
  Changes went through **Mini-PAM**, sized by how much the change moved expected
  performance. Portfolio managers then decided whether and how to use a model.
* From at least March 2019, employees warned that many staff had unreviewed
  read/write access to celFS. Between November 2021 and August 2023, one
  modeler ("Modeler A") made unauthorised changes to decorrelation parameters
  of **14 live models**, often setting them to zero.
* Impact: some funds overperformed by more than $400M and others
  underperformed by about $165M. Two Sigma repaid ~$165M and paid a **$90M**
  civil penalty ($45M each). Violations: Advisers Act §206(2), §206(4) and
  Rule 206(4)-7, failure to supervise, and Exchange Act Rule 21F-17(a)
  (separation agreements that could impede whistleblowing).

**Our inference**
* The architecture is "many forecast models → combination with explicit
  de-duplication → optimizer". That is the shape our pipeline copies.
* Setting a decorrelation parameter to zero most plausibly means "stop
  removing this model's overlap with others", which increases its effective
  weight in shared bets. The order does not give the formula; our
  `combine.decorrelate` strength parameter is one standard way to implement it.

## 2. TSI Form ADV Part 2A (March 2025)

**What it says**
* "Process-driven, systematic investment management using quantitative
  analysis"; strategies built on patterns in historical prices and other data,
  including "nontraditional (i.e., alternative) data".
* Strategy types listed: statistically based, merger/risk arbitrage,
  fundamentally driven, event driven, relative value, volatility arbitrage,
  structured credit, and contributor/sentiment-based (via a proprietary
  "alpha capture system").
* Uses machine learning and AI, including generative AI and LLMs.
* Strategies are combined into "quantitatively-optimized portfolios"; the
  optimizer produces a "target goal position" for each instrument and adjusts
  for risks tied to size, liquidity and sector exposure. Portfolio managers set
  leverage, turnover, position limits and overall risk.
* Risk is controlled "primarily … through a combination of strategy weightings,
  position limits", with controls built into each optimizer, and monitored with
  "Value at Risk (VaR) and similar calculations, stress-testing", model
  correlation metrics and factor measurements. A Chief Risk Officer acts as an
  independent check.
* Risk disclosures: simulations are hindsight-designed; strategies may decay;
  crowding with similar firms; frequent trading raises costs; borrow costs can
  be prohibitive; capacity limits (profitability expected to fall as assets grow).
* The brochure says the firm "does not expect to disclose discovered Process
  Errors" to clients and refers to protecting the "confidential and
  proprietary nature" of its strategies.

**Our inference**
* The process maps onto the textbook "alpha model → risk model → transaction
  cost model → portfolio construction → execution" stack (e.g. Narang,
  *Inside the Black Box*). The brochure confirms the stages; it gives none of
  the mathematics.

## 3. Two Sigma Investor Solutions (Venn) Form ADV Part 2A (March 2025)

**What it says**
* Venn analyses client portfolios with the **Two Sigma Factor Lens** using
  "optimization- and regression-based analysis", factor exposure, residual
  exposure, contributions to risk and return, drawdown, scenario and
  forward-looking analyses.
* It does not list the lens's factors or construction in the brochure.

**Where the detail actually is:** the 2018 Two Sigma research paper (source 6,
not an SEC filing) describes the lens: core factors Equity, Interest Rates,
Credit, Commodities; secondary factors Foreign Currency, Emerging Markets,
Equity Short Volatility, Local Inflation; credit and commodities residualised
on equity and rates; secondary factors residualised on all four core factors;
the residual keeps the intercept; three-year, exponentially weighted daily
regressions. `factor_lens.py` rebuilds this with ETF proxies.

## 4. Form 13F

**What it says:** quarterly long US equity (and listed option) positions, with
CUSIP, shares and value. No shorts, no futures/swaps, no model information.

**Use in this project:** `edgar.py` downloads the latest 13F as data — for a
point-in-time universe or crowding studies. It reveals what was held, not why.

## What is *not* public

Signal definitions, model weights, optimizer objective and constraints,
risk-model factors, covariance estimators, VaR confidence levels and horizons,
stress scenarios, transaction-cost models, or any parameter values.
Anything in this project at that level of detail is a standard published
technique, cited in each module's docstring.
