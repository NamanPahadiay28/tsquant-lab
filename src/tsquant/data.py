"""Data sourcing from free public sources, with a local CSV cache.

Sources
-------
* Prices / volume: Yahoo Finance via the ``yfinance`` package (optional extra).
* Fama-French daily factors + momentum: Kenneth French Data Library.
* Risk-free rate and macro series: FRED (St. Louis Fed) CSV endpoint.

Every loader caches to ``cache_dir`` so a backtest is reproducible offline
once the data has been fetched. Free data has known biases (survivorship in a
hand-picked universe, Yahoo adjustments that change over time); see
docs/WORKFLOW.md.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_daily_CSV.zip"
)
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


@dataclass
class MarketData:
    """Aligned daily panels. Columns are tickers, index is dates."""

    prices: pd.DataFrame
    volume: pd.DataFrame
    lens_prices: pd.DataFrame  # ETFs used to build the macro factor lens
    rf_daily: pd.Series  # daily risk-free rate (decimal)

    @property
    def returns(self) -> pd.DataFrame:
        return self.prices.pct_change(fill_method=None)


def _cache_path(cache_dir: str | Path, name: str) -> Path:
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{name}.csv"


def load_prices(
    tickers: list[str], start: str, end: str, cache_dir: str | Path, tag: str = "prices"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adjusted close and volume from Yahoo Finance, cached as CSV."""
    pc, vc = _cache_path(cache_dir, f"{tag}_close"), _cache_path(cache_dir, f"{tag}_volume")
    if pc.exists() and vc.exists():
        close = pd.read_csv(pc, index_col=0, parse_dates=True)
        vol = pd.read_csv(vc, index_col=0, parse_dates=True)
        if set(tickers) <= set(close.columns):
            return close[tickers].loc[start:end], vol[tickers].loc[start:end]
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover - depends on optional extra
        raise ImportError("pip install 'tsquant[data]' (needs yfinance) or run with --synthetic") from e
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="column")
    close = raw["Close"].reindex(columns=tickers)
    vol = raw["Volume"].reindex(columns=tickers)
    close.to_csv(pc)
    vol.to_csv(vc)
    return close, vol


def _parse_french_csv(text: str) -> pd.DataFrame:
    """Parse a Ken French CSV: keep only rows that start with a YYYYMMDD date."""
    lines = text.splitlines()
    header_idx = next(i for i, ln in enumerate(lines) if re.match(r"^\s*\d{8}\s*,", ln)) - 1
    cols = [c.strip() for c in lines[header_idx].split(",")]
    cols[0] = "date"
    rows = [ln for ln in lines[header_idx + 1 :] if re.match(r"^\s*\d{8}\s*,", ln)]
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None, names=cols)
    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), format="%Y%m%d")
    return df.set_index("date").astype(float) / 100.0  # percent -> decimal


def load_fama_french(cache_dir: str | Path) -> pd.DataFrame:
    """Daily FF5 factors + momentum (Mkt-RF, SMB, HML, RMW, CMA, RF, Mom)."""
    cp = _cache_path(cache_dir, "ff_daily")
    if cp.exists():
        return pd.read_csv(cp, index_col=0, parse_dates=True)
    frames = []
    for url in (FF5_URL, MOM_URL):
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            frames.append(_parse_french_csv(z.read(z.namelist()[0]).decode("latin-1")))
    ff = frames[0].join(frames[1], how="inner")
    ff.columns = [c.strip() for c in ff.columns]
    ff.to_csv(cp)
    return ff


def load_fred(series: str, cache_dir: str | Path) -> pd.Series:
    """A FRED series as a float Series (missing values dropped)."""
    cp = _cache_path(cache_dir, f"fred_{series}")
    if cp.exists():
        return pd.read_csv(cp, index_col=0, parse_dates=True).iloc[:, 0]
    df = pd.read_csv(FRED_URL.format(series=series), na_values=".")
    date_col = df.columns[0]  # "DATE" or "observation_date" depending on FRED version
    s = df.set_index(pd.to_datetime(df[date_col]))[series].astype(float).dropna()
    s.to_csv(cp)
    return s


def load_market_data(cfg: dict) -> MarketData:
    """Fetch (or read from cache) everything the pipeline needs."""
    d = cfg["data"]
    cache = d["cache_dir"]
    prices, volume = load_prices(d["universe"], d["start"], d["end"], cache, "universe")
    lens_tickers = sorted({t for v in cfg["factor_lens"]["instruments"].values() for t in _as_list(v)})
    lens_prices, _ = load_prices(lens_tickers, d["start"], d["end"], cache, "lens")
    tbill = load_fred("DTB3", cache) / 100.0  # 3m T-bill, annualised percent
    rf = (tbill / 252.0).reindex(prices.index).ffill().fillna(0.0)
    return clean(MarketData(prices, volume, lens_prices, rf), d.get("min_history_days", 252))


def clean(md: MarketData, min_history_days: int = 252) -> MarketData:
    """Drop tickers with too little history and forward-fill short gaps."""
    keep = md.prices.notna().sum() >= min_history_days
    prices = md.prices.loc[:, keep].ffill(limit=5)
    volume = md.volume.reindex(columns=prices.columns).fillna(0.0)
    lens = md.lens_prices.reindex(prices.index).ffill(limit=5)
    return MarketData(prices, volume, lens, md.rf_daily.reindex(prices.index).fillna(0.0))


def _as_list(v) -> list[str]:
    return v if isinstance(v, list) else [v]


def assert_no_future_dates(df: pd.DataFrame, asof: pd.Timestamp) -> None:
    """Guard used in tests: a panel sliced 'as of' a date must not exceed it."""
    if len(df) and df.index.max() > asof:
        raise AssertionError(f"look-ahead: data through {df.index.max()} used at {asof}")


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff()
