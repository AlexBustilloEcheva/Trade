import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from alex_quant.backtest.walk_forward import signal_dates, walk_forward
from alex_quant.data.base import DataError, trading_sessions
from alex_quant.features import build_features
from alex_quant.labels import build_labels


def test_purged_windows_and_global_training(bars, config, sessions):
    config = config.model_copy(deep=True)
    config.walk_forward.embargo_sessions = 2
    result = walk_forward(
        build_features(bars, config.universe),
        build_labels(bars, config.universe, config.research.horizon_sessions),
        sessions,
        config,
    )
    for window in result.training_windows:
        start = pd.Timestamp(window["prediction_start"])
        train = pd.DatetimeIndex(window["train_dates"])
        preds = result.predictions.loc[result.predictions.model_id == window["model_id"]]
        assert train.max() < start
        assert pd.Timestamp(window["label_end_max"]) < sessions[sessions.get_loc(start) - 2]
        assert set(train).isdisjoint(preds.signal_date)
        assert (
            config.walk_forward.min_train_sessions
            <= len(train)
            <= config.walk_forward.train_sessions
        )
        assert window["n_train_rows"] == len(train) * len(config.universe)
        assert preds.signal_date.nunique() <= config.walk_forward.retrain_every_signals
        assert (preds.execution_date > preds.signal_date).all()
        assert train.min() >= sessions[config.research.minimum_history_sessions]
    assert result.predictions.groupby("signal_date").selected.sum().eq(5).all()
    assert result.predictions.groupby("signal_date").target_weight.sum().eq(1).all()


def test_future_prices_cannot_change_historical_predictions(bars, config, sessions):
    cutoff = sessions[260]
    prefix = bars.loc[:cutoff]
    partial = walk_forward(
        build_features(prefix, config.universe),
        build_labels(prefix, config.universe, 10),
        sessions[sessions <= cutoff],
        config,
    )
    full = walk_forward(
        build_features(bars, config.universe),
        build_labels(bars, config.universe, 10),
        sessions,
        config,
    )
    common = full.predictions.loc[
        full.predictions.signal_date.isin(partial.predictions.signal_date)
    ]
    assert_frame_equal(partial.predictions, common.reset_index(drop=True), check_exact=True)


def test_weekly_holiday_and_incomplete_week():
    days = trading_sessions(pd.Timestamp("2024-03-25").date(), pd.Timestamp("2024-04-03").date())
    # Good Friday: Thursday is the completed week-end. Wednesday must not become a signal.
    assert list(signal_dates(days, "W-FRI")) == [pd.Timestamp("2024-03-28")]


@pytest.mark.parametrize("frequency", ["W-MON", "daily", "monthly"])
def test_other_rebalance_frequencies(sessions, frequency):
    signals = signal_dates(sessions, frequency)
    assert len(signals) > 0
    assert signals.is_unique and signals.is_monotonic_increasing
    assert signals.isin(sessions[:-1]).all()
    if frequency == "daily":
        assert len(signals) == len(sessions) - 1


def test_insufficient_history_is_an_error(bars, config, sessions):
    with pytest.raises(DataError, match="Histórico insuficiente"):
        walk_forward(
            build_features(bars.loc[: sessions[130]], config.universe),
            build_labels(bars.loc[: sessions[130]], config.universe, 10),
            sessions[:131],
            config,
        )
