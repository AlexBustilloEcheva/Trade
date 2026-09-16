import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from alex_quant.config import load_config
from alex_quant.data.base import DataError
from alex_quant.data.fixture import FixtureProvider
from alex_quant.data.snapshot import SnapshotProvider
from alex_quant.pipeline import run_pipeline


def test_full_pipeline_artifacts_replay_and_portfolio_constraints(config, tmp_path):
    output, replay = tmp_path / "first", tmp_path / "replay"
    metrics = run_pipeline(config, FixtureProvider(config.research.seed), output)
    required = ["metrics.json", "equity.csv", "predictions.csv", "positions.csv", "report.md"]
    assert all((output / name).is_file() for name in required)
    assert "DATOS SINTÉTICOS" in (output / "report.md").read_text(encoding="utf-8")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["source"]["synthetic"] is True
    for name, expected_hash in manifest["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected_hash
    assert set(metrics) == {"stockranker", "SPY", "equal_weight", "cash"}
    positions = pd.read_csv(output / "positions.csv")
    strategy = positions.loc[positions.portfolio == "stockranker"]
    assert strategy.groupby("date").size().max() == 5
    assert strategy.weight.ge(0).all()
    assert strategy.groupby("date").weight.sum().le(1 + 1e-12).all()
    predictions = pd.read_csv(output / "predictions.csv")
    assert predictions.groupby("signal_date").selected.sum().eq(5).all()
    equity = pd.read_csv(output / "equity.csv")
    for _, group in equity.groupby("portfolio"):
        assert len(group) == len(equity.date.unique())
        assert group.is_baseline.sum() == 1
    second = run_pipeline(load_config(output / "config.json"), SnapshotProvider(output), replay)
    assert metrics == second
    # Numerical artifacts must be byte-identical, including models and the full precision snapshot.
    for name in manifest["files"]:
        assert (output / name).read_bytes() == (replay / name).read_bytes(), name
    assert json.loads((replay / "manifest.json").read_text())["source"]["synthetic"] is True
    with (output / "input_bars.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(DataError, match="hash"):
        SnapshotProvider(output)


def test_fixture_is_deterministic_and_explicit(config):
    provider = FixtureProvider(config.research.seed)
    first = provider.fetch(config.symbols, config.data.start, config.data.end)
    second = provider.fetch(config.symbols, config.data.start, config.data.end)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert provider.provenance["synthetic"] is True
    assert Path(".env.example").read_text().count("KEY=\n") == 2
