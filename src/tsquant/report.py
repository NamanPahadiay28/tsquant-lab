"""Write run outputs: metrics.json, CSVs and a one-page tearsheet PNG."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tsquant.evaluate import drawdown  # noqa: E402

INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
NET, GROSS = "#2563eb", "#9ca3af"


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def write_outputs(res: dict, out_dir: str | Path, params_hash: str | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"params_hash": params_hash, **res["metrics"]}
    (out / "metrics.json").write_text(json.dumps(_jsonable(meta), indent=2))
    (out / "admission.json").write_text(json.dumps(_jsonable(res["admission"]), indent=2))
    res["daily"].to_csv(out / "daily.csv")
    res["weights"].to_csv(out / "weights.csv")
    res["ic_summary"].to_csv(out / "ic_summary.csv")
    res["signal_corr"].to_csv(out / "signal_corr.csv")
    res["lens_exposures"].to_csv(out / "lens_exposures.csv")
    res["style_exposures"].to_csv(out / "style_exposures.csv")
    pd.concat({"historical": res["stress_historical"], "factor_shock": res["stress_factor"]}).to_csv(out / "stress.csv")
    tearsheet(res, out / "tearsheet.png")
    return out


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_visible(False)


def tearsheet(res: dict, path: Path) -> None:
    d = res["daily"]
    m = res["metrics"]["net"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    ax = axes[0, 0]
    ax.plot((1 + d["gross"]).cumprod(), color=GROSS, lw=1.2, label="Gross")
    ax.plot((1 + d["net"]).cumprod(), color=NET, lw=1.6, label="Net of costs")
    ax.legend(frameon=False, fontsize=8)
    _style(ax, f"Growth of $1 · net Sharpe {m['sharpe']:.2f}, ann. vol {m['ann_vol']:.1%}")

    ax = axes[0, 1]
    dd = drawdown(d["net"])
    ax.fill_between(dd.index, dd, 0, color=NET, alpha=0.25, lw=0)
    ax.plot(dd, color=NET, lw=1)
    _style(ax, f"Drawdown · max {m['max_drawdown']:.1%}")

    ax = axes[1, 0]
    roll = d["net"].rolling(252).mean() / d["net"].rolling(252).std() * np.sqrt(252)
    ax.plot(roll, color=NET, lw=1.2)
    ax.axhline(0, color=MUTED, lw=0.8)
    _style(ax, "Rolling 1-year Sharpe (net)")

    ax = axes[1, 1]
    le = res["lens_exposures"].drop(index="alpha")
    ax.barh(le.index, le["coef"], color=NET)
    ax.axvline(0, color=MUTED, lw=0.8)
    for y, (c, t) in enumerate(zip(le["coef"], le["t_stat"])):
        ax.text(c, y, f"  t={t:.1f}", va="center", fontsize=7, color=MUTED)
    _style(ax, f"Macro factor lens betas · R² {res['lens_exposures'].attrs['r2']:.2f}")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
