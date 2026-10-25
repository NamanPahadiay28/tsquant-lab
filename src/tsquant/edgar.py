"""Pull Form 13F holdings from SEC EDGAR (free; requires a User-Agent header).

This is how the SEC filings become *data* in the project: a 13F lists a
manager's long US equity positions each quarter (no shorts, no derivatives
beyond listed options, filed up to 45 days after quarter end). Use it to build
a point-in-time universe or to study crowding; do not use the latest filing as
the universe for a historical backtest (that is look-ahead bias).

Two Sigma Investments, LP files under CIK 1179392. Since 2023 the "value"
column is in dollars; earlier filings report thousands of dollars.

SEC fair-access policy: identify yourself in User-Agent, max 10 requests/sec.
https://www.sec.gov/os/accessing-edgar-data
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/"
TWO_SIGMA_INVESTMENTS_CIK = 1179392


def _get(url: str, user_agent: str) -> requests.Response:
    r = requests.get(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}, timeout=60)
    r.raise_for_status()
    time.sleep(0.15)
    return r


def list_13f_filings(cik: int, user_agent: str) -> pd.DataFrame:
    js = _get(SUBMISSIONS.format(cik=cik), user_agent).json()
    rec = pd.DataFrame(js["filings"]["recent"])
    rec = rec[rec["form"].isin(["13F-HR", "13F-HR/A"])]
    return rec[["accessionNumber", "filingDate", "reportDate", "form"]].reset_index(drop=True)


def parse_information_table(xml_text: str) -> pd.DataFrame:
    """Parse a 13F information-table XML (namespace-agnostic)."""
    root = ET.fromstring(xml_text)

    def local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    rows = []
    for it in root.iter():
        if local(it.tag) != "infoTable":
            continue
        rec = {}
        for el in it.iter():
            name = local(el.tag)
            if name in {"nameOfIssuer", "titleOfClass", "cusip", "value", "sshPrnamt", "sshPrnamtType", "putCall", "investmentDiscretion"}:
                rec[name] = (el.text or "").strip()
        rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ("value", "sshPrnamt"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "putCall" not in df:
        df["putCall"] = ""
    return df


def fetch_13f_holdings(cik: int, accession: str, user_agent: str) -> pd.DataFrame:
    acc = accession.replace("-", "")
    index = _get(ARCHIVE.format(cik=cik, acc=acc) + "index.json", user_agent).json()
    xmls = [i["name"] for i in index["directory"]["item"] if i["name"].lower().endswith(".xml")]
    table = next((n for n in xmls if "primary_doc" not in n.lower()), None)
    if table is None:
        raise RuntimeError(f"no information table XML in {accession}: {xmls}")
    return parse_information_table(_get(ARCHIVE.format(cik=cik, acc=acc) + table, user_agent).text)


def top_holdings(h: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """Aggregate share lines by CUSIP (ignore option lines) and rank by value."""
    eq = h[h["putCall"].fillna("") == ""]
    agg = eq.groupby(["cusip", "nameOfIssuer"], as_index=False)[["value", "sshPrnamt"]].sum()
    agg["weight"] = agg["value"] / agg["value"].sum()
    return agg.sort_values("value", ascending=False).head(n).reset_index(drop=True)


def cusips_to_tickers(cusips: list[str], api_key: str | None = None) -> dict[str, str]:
    """Map CUSIPs to US tickers with the free OpenFIGI API (rate limited)."""
    url = "https://api.openfigi.com/v3/mapping"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-OPENFIGI-APIKEY"] = api_key
    out: dict[str, str] = {}
    batch = 100 if api_key else 10
    for i in range(0, len(cusips), batch):
        jobs = [{"idType": "ID_CUSIP", "idValue": c, "exchCode": "US"} for c in cusips[i : i + batch]]
        r = requests.post(url, json=jobs, headers=headers, timeout=60)
        r.raise_for_status()
        for c, res in zip(cusips[i : i + batch], r.json()):
            data = res.get("data") or []
            if data:
                out[c] = data[0]["ticker"]
        time.sleep(2.5 if not api_key else 0.3)
    return out
