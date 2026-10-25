"""Command line entry point.

  tsquant approve --reason "initial parameters"   # lock current parameters
  tsquant run [--synthetic]                        # full pipeline -> outputs/
  tsquant admit --candidate low_beta               # PAM-style correlation gate
  tsquant edgar-13f --user-agent "Name email"      # Two Sigma's latest 13F
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from tsquant.governance import ParamRegistry, UnapprovedParametersError, admission_test, model_params


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _registry(cfg: dict) -> ParamRegistry:
    g = cfg["governance"]
    return ParamRegistry(g["lock_file"], g["change_log"])


def _market(cfg: dict, synthetic: bool):
    if synthetic:
        from tsquant.synthetic import make_synthetic_market

        s = cfg.get("synthetic", {})
        return make_synthetic_market(s.get("n_stocks", 80), s.get("n_days", 2520), s.get("seed", 7))
    from tsquant.data import load_market_data

    return load_market_data(cfg)


def cmd_run(args) -> int:
    cfg = load_config(args.config)
    reg = _registry(cfg)
    params = model_params(cfg)
    try:
        h = reg.verify(params)
    except UnapprovedParametersError as e:
        if not args.allow_unapproved:
            print(f"refusing to run: {e}", file=sys.stderr)
            return 2
        logging.warning("running with UNAPPROVED parameters (research mode): %s", e)
        h = None
    from tsquant.pipeline import run_pipeline
    from tsquant.report import write_outputs

    md = _market(cfg, args.synthetic)
    res = run_pipeline(md, cfg)
    out = write_outputs(res, Path(cfg["output_dir"]) / ("synthetic" if args.synthetic else "real"), h)
    m = res["metrics"]["net"]
    print(f"net Sharpe {m['sharpe']:.2f} | ann ret {m['ann_return']:.2%} | vol {m['ann_vol']:.2%} | "
          f"maxDD {m['max_drawdown']:.2%} | DSR {res['metrics']['deflated_sharpe']:.2f}")
    print(f"outputs written to {out}")
    return 0


def cmd_approve(args) -> int:
    cfg = load_config(args.config)
    h = _registry(cfg).approve(model_params(cfg), args.reason, args.approver)
    print(f"approved parameters {h[:12]}")
    return 0


def cmd_admit(args) -> int:
    cfg = load_config(args.config)
    from tsquant.signals import compute_signals

    md = _market(cfg, args.synthetic)
    live = cfg["signals"]["names"]
    sig = compute_signals(md.prices, md.volume, sorted(set(live) | {args.candidate}))
    res = admission_test(args.candidate, sig, live, cfg["governance"]["max_abs_corr"])
    print(json.dumps(res, indent=2))
    return 0 if res["admitted"] else 1


def cmd_edgar(args) -> int:
    from tsquant import edgar

    filings = edgar.list_13f_filings(args.cik, args.user_agent)
    latest = filings.iloc[0]
    h = edgar.fetch_13f_holdings(args.cik, latest["accessionNumber"], args.user_agent)
    top = edgar.top_holdings(h, args.top)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(out, index=False)
    print(f"{latest['form']} for {latest['reportDate']} (filed {latest['filingDate']}): {len(h)} lines -> {out}")
    return 0


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="tsquant")
    p.add_argument("--config", default="configs/default.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the full pipeline")
    r.add_argument("--synthetic", action="store_true", help="use generated data (offline)")
    r.add_argument("--allow-unapproved", action="store_true", help="research mode: skip the parameter lock")
    r.set_defaults(fn=cmd_run)

    a = sub.add_parser("approve", help="approve the current model parameters")
    a.add_argument("--reason", required=True)
    a.add_argument("--approver")
    a.set_defaults(fn=cmd_approve)

    c = sub.add_parser("admit", help="correlation gate for a candidate signal")
    c.add_argument("--candidate", required=True)
    c.add_argument("--synthetic", action="store_true")
    c.set_defaults(fn=cmd_admit)

    e = sub.add_parser("edgar-13f", help="download a manager's latest 13F holdings")
    e.add_argument("--cik", type=int, default=1179392)
    e.add_argument("--user-agent", required=True, help='SEC requires "Your Name your@email"')
    e.add_argument("--top", type=int, default=50)
    e.add_argument("--out", default="data/13f_top_holdings.csv")
    e.set_defaults(fn=cmd_edgar)

    args = p.parse_args(argv)
    sys.exit(args.fn(args))
