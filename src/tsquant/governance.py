"""Model governance: parameter locking, change log, and a model admission test.

Grounded in the SEC's January 2025 order against Two Sigma (Admin. Proc. File
No. 3-22418). The order describes:

* "PAM" (Productionalize a Model): a white paper plus a form summarising the
  model's correlation to existing models; models could be approved if that
  correlation was below a threshold.
* "Mini-PAM" for changes to existing models, sized by how much the change
  moved expected performance.
* Model parameters stored in a database ("celFS") that many staff could edit
  without review; one modeler changed "decorrelation parameters" of fourteen
  live models without approval.

This module is our own small-scale version of the controls the order says were
missing. It is not Two Sigma's code or process.
"""

from __future__ import annotations

import getpass
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


class UnapprovedParametersError(RuntimeError):
    pass


def params_hash(params: dict) -> str:
    """Stable SHA-256 of the model parameters (key order independent)."""
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def model_params(cfg: dict) -> dict:
    """The subset of the config that changes trading behaviour."""
    keys = ("signals", "combination", "risk", "optimizer", "costs", "backtest")
    return {k: cfg[k] for k in keys if k in cfg}


class ParamRegistry:
    """Approve-then-run: a run refuses parameters that differ from the approved lock.

    ``approve`` writes the lock and appends a record (who, when, old/new hash,
    which keys changed, a free-text reason) to an append-only JSONL log.
    """

    def __init__(self, lock_path: str | Path, log_path: str | Path):
        self.lock_path, self.log_path = Path(lock_path), Path(log_path)

    def approved(self) -> dict | None:
        if not self.lock_path.exists():
            return None
        return json.loads(self.lock_path.read_text())

    def verify(self, params: dict) -> str:
        h = params_hash(params)
        lock = self.approved()
        if lock is None or lock["hash"] != h:
            changed = _diff_keys(lock["params"] if lock else {}, params)
            raise UnapprovedParametersError(
                f"parameters differ from approved lock (changed: {changed}). "
                "Review them, then run `tsquant approve --reason ...`."
            )
        return h

    def approve(self, params: dict, reason: str, approver: str | None = None) -> str:
        old = self.approved()
        h = params_hash(params)
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "approver": approver or getpass.getuser(),
            "reason": reason,
            "old_hash": old["hash"] if old else None,
            "new_hash": h,
            "changed": _diff_keys(old["params"] if old else {}, params),
        }
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.write_text(json.dumps({"hash": h, "params": params}, indent=2, default=str))
        with self.log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        return h


def _diff_keys(old: dict, new: dict, prefix: str = "") -> list[str]:
    out = []
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k), new.get(k)
        path = f"{prefix}{k}"
        if isinstance(a, dict) and isinstance(b, dict):
            out += _diff_keys(a, b, path + ".")
        elif a != b:
            out.append(path)
    return out


def mean_signal_correlation(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Average daily cross-sectional correlation between two signal panels."""
    a, b = a.align(b, join="inner")
    mask = a.notna() & b.notna()
    a, b = a.where(mask), b.where(mask)
    a = a.sub(a.mean(axis=1), axis=0)
    b = b.sub(b.mean(axis=1), axis=0)
    c = (a * b).sum(axis=1) / np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1))
    return float(c.dropna().mean())


def admission_test(
    candidate: str, signals: dict[str, pd.DataFrame], live: list[str], max_abs_corr: float
) -> dict:
    """PAM-style gate: admit a new signal only if its correlation to every live
    signal is below a threshold. Returns the correlations and the verdict."""
    corrs = {n: mean_signal_correlation(signals[candidate], signals[n]) for n in live if n != candidate}
    worst = max((abs(v) for v in corrs.values()), default=0.0)
    return {"candidate": candidate, "correlations": corrs, "max_abs_corr": worst, "admitted": worst < max_abs_corr}
