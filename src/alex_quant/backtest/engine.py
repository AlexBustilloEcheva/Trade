from dataclasses import dataclass

import pandas as pd

from alex_quant.config import Config
from alex_quant.portfolio import rebalance_after_costs


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    positions: pd.DataFrame
    trades: pd.DataFrame


def simulate(bars: pd.DataFrame, predictions: pd.DataFrame, config: Config) -> BacktestResult:
    opens, closes = bars.open.unstack("symbol"), bars.close.unstack("symbol")
    start = predictions.signal_date.min()
    sessions = closes.index[closes.index >= start]
    targets = {
        date: group.set_index("symbol").target_weight
        for date, group in predictions.groupby("execution_date", sort=True)
    }
    first_execution = min(targets)
    signal_by_execution = predictions.groupby("execution_date").signal_date.first()
    equity_rows, position_rows, trade_rows = [], [], []
    settings = config.portfolio
    daily_cash = (1 + settings.cash_annual_rate) ** (1 / settings.annualization) - 1
    for portfolio in ("stockranker", "SPY", "equal_weight", "cash"):
        holdings = pd.Series(0.0, index=closes.columns)
        cash = settings.initial_capital
        previous_equity = settings.initial_capital
        for i, session in enumerate(sessions):
            commission = slippage = turnover = 0.0
            rebalance = False
            if i:
                previous_close = closes.loc[sessions[i - 1]]
                # Overnight belongs to the old holdings; new holdings earn open-to-close only.
                holdings *= opens.loc[session] / previous_close
                cash *= 1 + daily_cash
                if portfolio in {"stockranker", "equal_weight"} and session in targets:
                    rebalance = True
                    weights = (
                        targets[session]
                        if portfolio == "stockranker"
                        else pd.Series(1 / len(config.universe), index=list(config.universe))
                    )
                elif portfolio == "SPY" and session == first_execution:
                    rebalance = True
                    weights = pd.Series({"SPY": 1.0})
                if rebalance:
                    pre_equity = float(holdings.sum() + cash)
                    holdings, cash, commission, slippage, trades = rebalance_after_costs(
                        holdings,
                        cash,
                        weights,
                        settings.commission_bps / 10000,
                        settings.slippage_bps / 10000,
                    )
                    turnover = float(trades.abs().sum()) / pre_equity
                    for symbol, notional in trades.items():
                        if abs(notional) > 1e-8:
                            trade_rows.append(
                                {
                                    "date": session,
                                    "signal_date": signal_by_execution.loc[session],
                                    "portfolio": portfolio,
                                    "symbol": symbol,
                                    "trade_notional": notional,
                                    "reference_open": opens.loc[session, symbol],
                                    "commission": abs(notional) * settings.commission_bps / 10000,
                                    "slippage": abs(notional) * settings.slippage_bps / 10000,
                                }
                            )
                holdings *= closes.loc[session] / opens.loc[session]
            equity = float(holdings.sum() + cash)
            equity_rows.append(
                {
                    "date": session,
                    "portfolio": portfolio,
                    "equity": equity,
                    "daily_return": equity / previous_equity - 1 if i else 0.0,
                    "turnover": turnover,
                    "cost": commission + slippage,
                    "commission": commission,
                    "slippage": slippage,
                    "rebalanced": rebalance,
                    "exposure": float(holdings.sum()) / equity,
                    "cash": cash,
                    "is_baseline": i == 0,
                }
            )
            for symbol, value in holdings.items():
                if value > 1e-8:
                    position_rows.append(
                        {
                            "date": session,
                            "portfolio": portfolio,
                            "symbol": symbol,
                            "weight": value / equity,
                            "notional": value,
                        }
                    )
            previous_equity = equity
    return BacktestResult(
        pd.DataFrame(equity_rows).sort_values(["date", "portfolio"]).reset_index(drop=True),
        pd.DataFrame(position_rows),
        pd.DataFrame(trade_rows),
    )
