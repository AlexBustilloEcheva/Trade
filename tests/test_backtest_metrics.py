import numpy as np
import pandas as pd
import pytest

from alex_quant.backtest.engine import simulate
from alex_quant.config import AllocationConfig, Config
from alex_quant.evaluation import calculate_metrics
from alex_quant.portfolio import allocate, rank_scores, rebalance_after_costs


def small_backtest(commission=0, slippage=0):
    config = Config.model_validate(
        {
            "data": {"start": "2024-01-02", "end": "2024-01-04"},
            "universe": {"A": "XLK", "B": "XLK"},
            "research": {"top_k": 1},
            "allocation": {"max_asset_weight": 1.0, "max_sector_weight": 1.0},
            "portfolio": {
                "initial_capital": 100,
                "commission_bps": commission,
                "slippage_bps": slippage,
            },
        }
    )
    dates = pd.date_range("2024-01-02", periods=3, freq="B", name="date")
    records = []
    for i, date in enumerate(dates):
        for symbol in ["A", "B", "SPY"]:
            opening = ([100, 200, 220] if symbol == "A" else [100, 100, 50])[i]
            close = ([100, 220, 242] if symbol == "A" else [100, 100, 55])[i]
            records.append({"date": date, "symbol": symbol, "open": opening, "close": close})
    bars = pd.DataFrame(records).set_index(["date", "symbol"]).sort_index()
    predictions = pd.DataFrame(
        [
            {
                "signal_date": dates[0],
                "execution_date": dates[1],
                "symbol": "A",
                "target_weight": 1.0,
            },
            {
                "signal_date": dates[1],
                "execution_date": dates[2],
                "symbol": "B",
                "target_weight": 1.0,
            },
        ]
    )
    return simulate(bars, predictions, config)


def test_trades_use_next_open_and_old_holdings_get_overnight_return():
    result = small_backtest()
    strategy = result.equity.loc[result.equity.portfolio == "stockranker"]
    # No windfall from A's overnight doubling before our first purchase, or B's pre-purchase crash.
    assert strategy.equity.to_list() == pytest.approx([100, 110, 121])
    assert strategy.exposure.to_list() == pytest.approx([0, 1, 1])
    assert strategy.turnover.to_list() == pytest.approx([0, 1, 2])
    assert (result.trades.date > result.trades.signal_date).all()
    positions = result.positions.loc[result.positions.portfolio == "stockranker"]
    assert positions.groupby("date").size().max() == 1


def test_costs_reduce_equity_and_reconcile_to_trades():
    free, costly = small_backtest(), small_backtest(10, 20)
    for name in ["stockranker", "SPY", "equal_weight"]:
        net = costly.equity.loc[costly.equity.portfolio == name]
        gross = free.equity.loc[free.equity.portfolio == name]
        assert net.equity.iloc[-1] < gross.equity.iloc[-1]
        assert net.cost.sum() > 0
        trades = costly.trades.loc[costly.trades.portfolio == name]
        assert net.commission.sum() == pytest.approx(trades.trade_notional.abs().sum() * 0.001)
        assert net.slippage.sum() == pytest.approx(trades.trade_notional.abs().sum() * 0.002)
        assert net.cash.min() >= 0
        assert net.exposure.max() <= 1 + 1e-12
    assert costly.equity.loc[costly.equity.portfolio == "cash", "equity"].eq(100).all()


def test_costs_and_drift_use_actual_holdings():
    holdings = pd.Series({"A": 70.0, "B": 30.0})
    desired, cash, commission, slippage, trades = rebalance_after_costs(
        holdings,
        0,
        pd.Series({"A": 0.5, "B": 0.5}),
        0.001,
        0.002,
    )
    assert desired.A == pytest.approx(desired.B)
    assert desired.sum() + cash + commission + slippage == pytest.approx(100)
    assert trades.abs().sum() == pytest.approx(40)
    assert commission + slippage == pytest.approx(0.12)


def test_rank_ties_are_deterministic_and_limited_to_top_five():
    ranking = allocate(
        rank_scores(pd.DataFrame({"symbol": list("GFEDCBA"), "score": [1.0] * 7})),
        {symbol: symbol for symbol in "GFEDCBA"},
        AllocationConfig(),
    )
    assert ranking.loc[ranking.selected, "symbol"].tolist() == list("ABCDE")
    assert ranking.loc[ranking.selected, "target_weight"].to_list() == pytest.approx([0.2] * 5)


def test_no_shorting_or_leverage():
    for weights in [pd.Series({"A": 1.1}), pd.Series({"A": -0.1})]:
        with pytest.raises(ValueError, match="sin apalancamiento"):
            rebalance_after_costs(pd.Series({"A": 0.0}), 100, weights, 0, 0)


def curve(values):
    n = len(values)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="B"),
            "equity": values,
            "cost": [0.0] * n,
            "commission": [0.0] * n,
            "slippage": [0.0] * n,
            "turnover": [0.0, 1.0] + [0.0] * (n - 2),
            "rebalanced": [False, True] + [False] * (n - 2),
            "exposure": [0.0] + [1.0] * (n - 1),
        }
    )


def test_metrics_against_hand_calculation():
    m = calculate_metrics(curve([100, 110, 99, 108.9]), annualization=3)
    returns = np.array([0.1, -0.1, 0.1])
    assert m["cumulative_return"] == pytest.approx(0.089)
    assert m["annualized_return"] == pytest.approx(0.089)
    assert m["annualized_volatility"] == pytest.approx(returns.std(ddof=1) * np.sqrt(3))
    assert m["sharpe"] == pytest.approx(returns.mean() / returns.std(ddof=1) * np.sqrt(3))
    assert m["sortino"] == pytest.approx(1.0)
    assert m["max_drawdown"] == pytest.approx(-0.1)
    assert m["calmar"] == pytest.approx(0.89)
    assert m["turnover_annualized"] == pytest.approx(1.0)
    assert m["rebalance_count"] == 1
    assert m["average_exposure"] == 1


def test_initial_capital_counts_in_drawdown_and_undefined_ratios_are_null():
    m = calculate_metrics(curve([100, 90, 81]))
    assert m["max_drawdown"] == pytest.approx(-0.19)
    cash = calculate_metrics(curve([100, 100, 100]))
    assert cash["annualized_return"] == 0
    assert cash["annualized_volatility"] == 0
    assert cash["sharpe"] is cash["sortino"] is cash["calmar"] is None
