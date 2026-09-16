from datetime import date

import httpx
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pydantic import ValidationError

from alex_quant.backtest.benchmarks import momentum_predictions
from alex_quant.backtest.engine import simulate
from alex_quant.backtest.walk_forward import walk_forward
from alex_quant.cli import main
from alex_quant.config import AllocationConfig, Config
from alex_quant.data.alpaca import AlpacaProvider
from alex_quant.data.base import DataError, trading_sessions
from alex_quant.data.diagnostics import diagnose_data
from alex_quant.data.schedule import rebalance_schedule
from alex_quant.evaluation.research import rank_information_coefficient, yearly_metrics
from alex_quant.features import build_features
from alex_quant.labels import build_labels
from alex_quant.portfolio import allocate, rank_scores


def aligned(bars, config):
    sessions = bars.index.get_level_values("date").unique()
    schedule = rebalance_schedule(sessions, config.research.rebalance_frequency)
    labels = build_labels(bars, config.universe, mode="rebalance_open_to_open", schedule=schedule)
    features = build_features(bars, config.universe)
    result = walk_forward(features, labels, sessions, config, schedule)
    return result, labels, features, schedule


def test_open_labels_holidays_missing_final_exit_and_no_artificial_cutoff(bars, config):
    _, labels, _, schedule = aligned(bars, config)
    signal = pd.Timestamp("2024-03-28")  # Good Friday 29th is a holiday.
    entry, end = pd.Timestamp("2024-04-01"), pd.Timestamp("2024-04-08")
    row = labels.loc[(signal, "AAPL")]
    assert row.entry_date == entry and row.label_end == end
    opens = bars.open.unstack("symbol")
    expected = (
        opens.loc[end, "AAPL"] / opens.loc[entry, "AAPL"]
        - opens.loc[end, "XLK"] / opens.loc[entry, "XLK"]
    )
    assert row.target == pytest.approx(expected)
    last = labels.xs(schedule.index[-1], level="date")
    assert last.target.isna().all() and last.label_end.isna().all()
    assert last.entry_date.notna().all()
    assert schedule.index[-1] == pd.Timestamp("2024-04-26")
    assert pd.Timestamp("2024-04-30") not in schedule.index
    assert labels.loc[labels.index.get_level_values("date").dayofweek == 1, "target"].isna().all()


def test_gap_before_entry_is_absent_from_target_and_new_position():
    sessions = trading_sessions(date(2024, 3, 22), date(2024, 4, 1))
    schedule = rebalance_schedule(sessions, "W-FRI")
    records = []
    for day in sessions:
        for symbol in ["A", "XLK", "SPY"]:
            price = 100.0
            if symbol == "A" and day >= pd.Timestamp("2024-03-25"):
                price = 200.0 if day < pd.Timestamp("2024-04-01") else 220.0
            records.append({"date": day, "symbol": symbol, "open": price, "close": price})
    bars = pd.DataFrame(records).set_index(["date", "symbol"]).sort_index()
    labels = build_labels(bars, {"A": "XLK"}, mode="rebalance_open_to_open", schedule=schedule)
    assert labels.loc[(sessions[0], "A"), "target"] == pytest.approx(0.1)
    config = Config.model_validate(
        {
            "data": {"start": "2024-03-22", "end": "2024-04-01"},
            "universe": {"A": "XLK"},
            "allocation": {"top_k": 1, "max_asset_weight": 1.0, "max_sector_weight": 1.0},
            "portfolio": {"initial_capital": 100, "commission_bps": 0, "slippage_bps": 0},
        }
    )
    predictions = schedule.reset_index().rename(columns={"entry_date": "execution_date"})
    predictions["symbol"], predictions["target_weight"] = "A", 1.0
    result = simulate(bars, predictions, config, schedule=schedule)
    curve = result.equity.loc[result.equity.portfolio == "stockranker"]
    assert curve.equity.iloc[1] == pytest.approx(100)
    assert curve.equity.iloc[-1] == pytest.approx(110)


def test_aligned_purge_and_future_invariance(bars, config, sessions):
    config = config.model_copy(deep=True)
    config.walk_forward.embargo_sessions = 2
    full, labels, _, _ = aligned(bars, config)
    for window in full.training_windows:
        cutoff = pd.Timestamp(window["label_cutoff_exclusive"])
        assert pd.Timestamp(window["label_end_max"]) < cutoff
        assert cutoff == sessions[sessions.get_loc(pd.Timestamp(window["prediction_start"])) - 2]
        dates = pd.to_datetime(window["train_dates"])
        actual = labels.loc[labels.index.get_level_values("date").isin(dates)]
        assert actual.target.notna().all()
        assert (actual.label_end < cutoff).all()
        assert (actual.entry_date > actual.index.get_level_values("date")).all()
    cutoff = sessions[260]
    partial, _, _, _ = aligned(bars.loc[:cutoff], config)
    common = full.predictions.loc[
        full.predictions.signal_date.isin(partial.predictions.signal_date)
    ]
    assert_frame_equal(partial.predictions, common.reset_index(drop=True), check_exact=True)
    assert full.predictions.signal_date.max() == pd.Timestamp("2024-04-26")
    assert labels.xs(full.predictions.signal_date.max(), level="date").target.isna().all()


def test_allocator_skips_full_sectors_and_does_not_filter_negative_scores():
    universe = {"A": "XLK", "B": "XLK", "C": "XLK", "D": "XLF", "E": "XLF", "F": "XLE"}
    ranking = rank_scores(
        pd.DataFrame({"symbol": list(universe), "score": [-1, -2, -3, -4, -5, -6]})
    )
    result = allocate(ranking, universe, AllocationConfig())
    assert result.loc[result.selected, "symbol"].to_list() == ["A", "B", "D", "E", "F"]
    assert result.target_weight.max() <= 0.2
    assert result.groupby("sector_etf").target_weight.sum().max() <= 0.4
    assert result.target_weight.sum() == pytest.approx(1)
    concentrated = allocate(ranking, dict.fromkeys(universe, "XLK"), AllocationConfig())
    assert concentrated.selected.sum() == 2
    assert 1 - concentrated.target_weight.sum() == pytest.approx(0.6)


@pytest.mark.parametrize("count", [1, 3, 5, 10])
def test_allocation_position_cap_and_residual_cash(count):
    universe = dict.fromkeys("ABCDE", "XLK")
    ranking = rank_scores(pd.DataFrame({"symbol": list(universe), "score": range(5)}))
    result = allocate(ranking, universe, AllocationConfig(top_k=count))
    assert result.selected.sum() <= min(count, len(universe))
    assert result.target_weight.max() <= 0.2
    assert result.target_weight.sum() <= 0.4 + 1e-12


def test_partial_sector_slot_is_not_redistributed():
    ranking = rank_scores(pd.DataFrame({"symbol": list("ABC"), "score": [3, 2, 1]}))
    result = allocate(ranking, dict.fromkeys("ABC", "XLK"), AllocationConfig(max_sector_weight=0.3))
    assert result.target_weight.to_list() == pytest.approx([0.2, 0.1, 0])


def test_momentum_uses_only_known_return_and_same_allocator(bars, config, sessions):
    result, _, features, _ = aligned(bars, config)
    momentum = momentum_predictions(features, result.predictions, config)
    first = momentum.signal_date.min()
    close = bars.close.unstack("symbol")
    lookback = sessions[sessions.get_loc(first) - 120]
    expected = close.loc[first, "AAPL"] / close.loc[lookback, "AAPL"] - 1
    row = momentum.loc[(momentum.signal_date == first) & (momentum.symbol == "AAPL")].iloc[0]
    assert row.score == pytest.approx(expected)
    assert set(momentum.signal_date) == set(result.predictions.signal_date)
    cutoff = sessions[260]
    past_predictions = result.predictions.loc[result.predictions.signal_date <= cutoff]
    past = momentum_predictions(
        build_features(bars.loc[:cutoff], config.universe), past_predictions, config
    )
    assert_frame_equal(
        past, momentum.loc[momentum.signal_date <= cutoff].reset_index(drop=True), check_exact=True
    )
    assert momentum.groupby(["signal_date", "sector_etf"]).target_weight.sum().max() <= 0.4 + 1e-12


def test_stress_keeps_predictions_and_applies_double_cost_rates(bars, config):
    result, _, features, schedule = aligned(bars, config)
    predictions_before = result.predictions.copy(deep=True)
    momentum = momentum_predictions(features, result.predictions, config)
    base = simulate(bars, result.predictions, config, momentum, schedule)
    stress = simulate(bars, result.predictions, config, momentum, schedule, cost_multiplier=2)
    assert_frame_equal(predictions_before, result.predictions)
    for portfolio in ["stockranker", "momentum", "SPY", "equal_weight"]:
        net = stress.equity.loc[stress.equity.portfolio == portfolio]
        normal = base.equity.loc[base.equity.portfolio == portfolio]
        assert net.equity.iloc[-1] < normal.equity.iloc[-1]
        assert net.date.to_list() == normal.date.to_list()
        assert net.rebalanced.to_list() == normal.rebalanced.to_list()
    expected_commission = (
        stress.trades.trade_notional.abs() * 2 * config.portfolio.commission_bps / 10000
    )
    np.testing.assert_allclose(stress.trades.commission, expected_commission)
    expected_slippage = (
        stress.trades.trade_notional.abs() * 2 * config.portfolio.slippage_bps / 10000
    )
    np.testing.assert_allclose(stress.trades.slippage, expected_slippage)


def test_daily_drift_is_reported_without_forced_daily_rebalancing(bars, config):
    constrained = config.model_copy(deep=True)
    constrained.universe = dict.fromkeys(["AAPL", "MSFT", "NVDA"], "XLK")
    result, _, _, schedule = aligned(bars, constrained)
    backtest = simulate(bars, result.predictions, constrained, schedule=schedule)
    equity = backtest.equity.loc[backtest.equity.portfolio == "stockranker"]
    assert equity.cash_weight.iloc[1] > 0.5
    assert equity.asset_limit_excess.max() > 0
    assert equity.sector_limit_excess.max() > 0
    assert equity.rebalanced.sum() == result.predictions.signal_date.nunique()
    assert (equity.loc[~equity.rebalanced, "cost"] == 0).all()
    assert (
        backtest.positions.loc[backtest.positions.portfolio == "stockranker"]
        .groupby("date")
        .size()
        .max()
        == 2
    )


def test_yearly_returns_and_drawdowns_include_first_loss():
    curve = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-12-28", "2023-12-29", "2024-01-02", "2024-01-03"]),
            "portfolio": "test",
            "daily_return": [0, 0.1, -0.1, 0.05],
            "is_baseline": [True, False, False, False],
        }
    )
    result = yearly_metrics(curve).set_index("year")
    assert result.loc[2023, "return"] == pytest.approx(0.1)
    assert result.loc[2024, "return"] == pytest.approx(-0.055)
    assert result.loc[2024, "max_drawdown"] == pytest.approx(-0.1)


def test_spearman_excludes_unmatured_labels_and_handles_ties():
    dates = pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-19"])
    predictions = pd.DataFrame(
        [
            {"signal_date": day, "symbol": symbol, "score": score}
            for day in dates
            for symbol, score in zip("ABC", [1, 2, 2], strict=True)
        ]
    )
    labels = pd.DataFrame(
        [
            {
                "date": day,
                "symbol": symbol,
                "target": target,
                "label_end": day + pd.Timedelta(days=7),
            }
            for day in dates
            for symbol, target in zip("ABC", [3, 1, 1], strict=True)
        ]
    ).set_index(["date", "symbol"])
    result = rank_information_coefficient(predictions, labels, dates[-1])
    assert result.signal_date.to_list() == list(dates[:2])
    assert result.spearman_ic.to_list() == pytest.approx([-1, -1])
    predictions["score"] = 0.0
    assert rank_information_coefficient(predictions, labels, dates[-1]).spearman_ic.isna().all()


def test_legacy_configuration_migration_is_unambiguous(config):
    value = config.model_dump(mode="json")
    value["research"]["top_k"] = 3
    with pytest.raises(ValidationError, match="solo allocation.top_k"):
        Config.model_validate(value)
    del value["allocation"]
    assert Config.model_validate(value).allocation.top_k == 3


def test_diagnostic_uses_only_spy_data_endpoint_and_never_prints_credentials(monkeypatch, capsys):
    monkeypatch.setenv("ALPACA_API_KEY", "diagnostic-test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "diagnostic-test-secret")
    loaded = []
    monkeypatch.setattr("alex_quant.cli.load_dotenv", lambda *a, **kw: loaded.append(True))
    requested = []

    def handler(request):
        requested.append(request)
        assert request.url.host == "data.alpaca.markets"
        assert request.url.path == "/v2/stocks/bars"
        assert request.url.params["symbols"] == "SPY"
        assert request.url.params["feed"] == "sip"
        records = [
            {"t": f"2024-01-0{day}T05:00:00Z", "o": 100, "h": 101, "l": 99, "c": 100, "v": 1000}
            for day in [2, 3, 4, 5]
        ]
        return httpx.Response(200, json={"bars": {"SPY": records}})

    monkeypatch.setattr(
        "alex_quant.cli.AlpacaProvider",
        lambda feed: AlpacaProvider(feed, httpx.MockTransport(handler)),
    )
    assert main(["diagnose-data", "--start", "2024-01-02", "--end", "2024-01-05"]) == 0
    output = capsys.readouterr().out
    assert '"bars": 4' in output and '"feed": "sip"' in output
    assert "diagnostic-test" not in output
    assert len(requested) == 1 and loaded == [True]


def test_diagnostic_rejects_large_intervals_before_loading_credentials(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("No debe cargar claves")

    monkeypatch.setattr("alex_quant.cli.load_dotenv", forbidden)
    assert main(["diagnose-data", "--start", "2020-01-01", "--end", "2025-01-01"]) == 2
    assert "diagnóstico" in capsys.readouterr().err


def test_diagnostic_missing_credentials(monkeypatch, capsys):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.setattr("alex_quant.cli.load_dotenv", lambda *a, **kw: None)
    assert main(["diagnose-data", "--start", "2024-01-02", "--end", "2024-01-05"]) == 2
    assert "ALPACA_API_KEY" in capsys.readouterr().err


def test_diagnostic_validates_response_dates_and_ohlcv(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test")

    def handler(request):
        return httpx.Response(
            200,
            json={
                "bars": {
                    "SPY": [
                        {
                            "t": "2024-01-02T05:00:00Z",
                            "o": 100,
                            "h": 101,
                            "l": 99,
                            "c": 100,
                            "v": 1000,
                        }
                    ]
                }
            },
        )

    provider = AlpacaProvider(transport=httpx.MockTransport(handler))
    with pytest.raises(DataError, match="ausentes"):
        diagnose_data(provider, date(2024, 1, 2), date(2024, 1, 5))


def test_fixture_cli_does_not_load_credentials(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Un fixture no debe leer .env")

    monkeypatch.setattr("alex_quant.cli.load_dotenv", forbidden)
    monkeypatch.setattr(
        "alex_quant.cli.run_pipeline", lambda *args: {"stockranker": {"rebalance_count": 1}}
    )
    assert (
        main(
            [
                "run",
                "--config",
                "configs/fixture.toml",
                "--provider",
                "fixture",
                "--output",
                str(tmp_path),
            ]
        )
        == 0
    )
