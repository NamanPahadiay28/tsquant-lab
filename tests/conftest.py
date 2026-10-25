import copy
from pathlib import Path

import pytest
import yaml

from tsquant.synthetic import make_synthetic_market

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def md():
    return make_synthetic_market(n_stocks=40, n_days=900, seed=3)


@pytest.fixture()
def cfg(tmp_path):
    c = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    c = copy.deepcopy(c)
    c["governance"]["lock_file"] = str(tmp_path / "lock.json")
    c["governance"]["change_log"] = str(tmp_path / "changes.jsonl")
    c["output_dir"] = str(tmp_path / "out")
    c["risk"]["window"] = 252
    c["factor_lens"]["window_days"] = 252
    c["combination"]["ic_lookback"] = 126
    return c
