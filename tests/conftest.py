from pathlib import Path

import pandas as pd
import pytest

from alex_quant.config import load_config
from alex_quant.data.fixture import FixtureProvider


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Las pruebas no pueden abrir conexiones de red")

    monkeypatch.setattr("socket.socket.connect", blocked)


@pytest.fixture(scope="session")
def config():
    return load_config(Path(__file__).resolve().parents[1] / "configs/fixture.toml")


@pytest.fixture(scope="session")
def bars(config):
    return FixtureProvider(config.research.seed).fetch(
        config.symbols, config.data.start, config.data.end
    )


@pytest.fixture(scope="session")
def sessions(bars):
    return pd.DatetimeIndex(bars.index.get_level_values("date").unique())
