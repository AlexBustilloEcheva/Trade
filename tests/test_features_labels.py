import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from alex_quant.features import build_features
from alex_quant.labels import build_labels


def test_features_never_change_when_future_is_added_or_mutated(bars, config, sessions):
    cutoff = sessions[210]
    dates = bars.index.get_level_values("date")
    past = bars.loc[dates <= cutoff]
    expected = build_features(past, config.universe)
    assert_frame_equal(expected, build_features(bars, config.universe).loc[:cutoff])
    poisoned = bars.copy()
    poisoned.loc[dates > cutoff, ["open", "high", "low", "close"]] *= 17
    poisoned.loc[dates > cutoff, "volume"] *= 93
    assert_frame_equal(expected, build_features(poisoned, config.universe).loc[:cutoff])


def test_features_are_invariant_to_uniform_later_adjustment(bars, config):
    adjusted = bars.copy()
    adjusted[["open", "high", "low", "close"]] *= 0.25
    adjusted["volume"] *= 4
    assert_frame_equal(
        build_features(bars, config.universe), build_features(adjusted, config.universe)
    )


def test_sector_and_spy_relative_strength_and_beta(bars, config, sessions):
    features = build_features(bars, config.universe)
    close = bars.close.unstack("symbol")
    day, old = sessions[180], sessions[160]
    stock_return = close.loc[day, "AAPL"] / close.loc[old, "AAPL"] - 1
    sector_return = close.loc[day, "XLK"] / close.loc[old, "XLK"] - 1
    spy_return = close.loc[day, "SPY"] / close.loc[old, "SPY"] - 1
    row = features.loc[(day, "AAPL")]
    assert row.relative_sector_20 == pytest.approx(stock_return - sector_return)
    assert row.relative_spy_20 == pytest.approx(stock_return - spy_return)
    returns = close.pct_change(fill_method=None).loc[:day].tail(60)
    beta = np.cov(returns.AAPL, returns.SPY, ddof=1)[0, 1] / returns.SPY.var(ddof=1)
    assert row.beta_spy_60 == pytest.approx(beta)
    volume = bars.volume.unstack("symbol")
    assert row.relative_volume_20 == pytest.approx(
        volume.loc[day, "AAPL"] / volume.AAPL.iloc[160:180].mean()
    )
    assert np.isnan(features.loc[(sessions[119], "AAPL"), "return_120"])


@pytest.mark.parametrize("horizon", [1, 10, 20])
def test_labels_correct_horizon_sector_and_unavailable_tail(bars, config, sessions, horizon):
    labels = build_labels(bars, config.universe, horizon)
    close = bars.close.unstack("symbol")
    day, future = sessions[150], sessions[150 + horizon]
    expected = (
        close.loc[future, "NVDA"] / close.loc[day, "NVDA"]
        - close.loc[future, "XLK"] / close.loc[day, "XLK"]
    )
    assert labels.loc[(day, "NVDA"), "target"] == pytest.approx(expected)
    assert labels.loc[(day, "NVDA"), "label_end"] == future
    tail = labels.loc[labels.index.get_level_values("date").isin(sessions[-horizon:])]
    assert tail.target.isna().all()
    assert tail.label_end.isna().all()
    mature = labels.loc[labels.index.get_level_values("date").isin(sessions[:-horizon])]
    assert mature.notna().all().all()
