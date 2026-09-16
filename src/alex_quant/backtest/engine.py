from dataclasses import dataclass

import pandas as pd

from alex_quant.config import Config
from alex_quant.portfolio import rebalance_after_costs


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    positions: pd.DataFrame
    trades: pd.DataFrame
    sector_exposure: pd.DataFrame


def simulate(
    bars: pd.DataFrame,
    predictions: pd.DataFrame,
    config: Config,
    momentum: pd.DataFrame | None = None,
    schedule: pd.DataFrame | None = None,
    cost_multiplier: float = 1.0,
) -> BacktestResult:
    if cost_multiplier <= 0:
        raise ValueError("El multiplicador de costes debe ser positivo")
    if schedule is not None:
        for frame in [predictions] + ([momentum] if momentum is not None else []):
            expected = frame.signal_date.map(schedule.entry_date)
            if not expected.equals(frame.execution_date):
                raise ValueError("Las ejecuciones no coinciden con el calendario compartido")
    opens, closes = bars.open.unstack("symbol"), bars.close.unstack("symbol")
    start = predictions.signal_date.min()
    sessions = closes.index[closes.index >= start]
    targets = {
        date: group.set_index("symbol").target_weight
        for date, group in predictions.groupby("execution_date", sort=True)
    }
    first_execution = min(targets)
    momentum_targets = (
        {}
        if momentum is None
        else {
            date: group.set_index("symbol").target_weight
            for date, group in momentum.groupby("execution_date", sort=True)
        }
    )
    if momentum is not None and set(momentum_targets) != set(targets):
        raise ValueError("Momentum y modelo deben compartir fechas de ejecución")
    signal_by_execution = predictions.groupby("execution_date").signal_date.first()
    equity_rows, position_rows, trade_rows, sector_rows = [], [], [], []
    settings = config.portfolio
    daily_cash = (1 + settings.cash_annual_rate) ** (1 / settings.annualization) - 1
    portfolios = ["stockranker", "SPY", "equal_weight", "cash"]
    if momentum is not None:
        portfolios.append("momentum")
    for portfolio in portfolios:
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
                if portfolio in {"stockranker", "equal_weight", "momentum"} and session in targets:
                    rebalance = True
                    weights = (
                        targets[session]
                        if portfolio == "stockranker"
                        else pd.Series(1 / len(config.universe), index=list(config.universe))
                    )
                    if portfolio == "momentum":
                        weights = momentum_targets[session]
                    if portfolio in {"stockranker", "momentum"}:
                        sector_weights = weights.groupby(weights.index.map(config.universe)).sum()
                        if (
                            (weights > config.allocation.max_asset_weight + 1e-12).any()
                            or (sector_weights > config.allocation.max_sector_weight + 1e-12).any()
                            or (weights > 0).sum() > config.allocation.top_k
                        ):
                            raise ValueError("La asignación incumple los límites de cartera")
                elif portfolio == "SPY" and session == first_execution:
                    rebalance = True
                    weights = pd.Series({"SPY": 1.0})
                if rebalance:
                    pre_equity = float(holdings.sum() + cash)
                    holdings, cash, commission, slippage, trades = rebalance_after_costs(
                        holdings,
                        cash,
                        weights,
                        cost_multiplier * settings.commission_bps / 10000,
                        cost_multiplier * settings.slippage_bps / 10000,
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
                                    "commission": abs(notional)
                                    * settings.commission_bps
                                    / 10000
                                    * cost_multiplier,
                                    "slippage": abs(notional)
                                    * settings.slippage_bps
                                    / 10000
                                    * cost_multiplier,
                                }
                            )
                holdings *= closes.loc[session] / opens.loc[session]
            equity = float(holdings.sum() + cash)
            weights_close = holdings / equity
            sectors = weights_close.groupby(
                [config.universe.get(symbol, f"BENCHMARK_{symbol}") for symbol in holdings.index]
            ).sum()
            max_asset, max_sector = float(weights_close.max()), float(sectors.max())
            constrained = portfolio in {"stockranker", "momentum"}
            for sector, weight in sectors.items():
                sector_rows.append(
                    {
                        "date": session,
                        "portfolio": portfolio,
                        "sector_etf": sector,
                        "weight": weight,
                    }
                )
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
                    "cash_weight": cash / equity,
                    "max_asset_weight": max_asset,
                    "max_sector_weight": max_sector,
                    "limits_applicable": constrained,
                    "asset_limit_excess": max(0, max_asset - config.allocation.max_asset_weight)
                    if constrained
                    else 0.0,
                    "sector_limit_excess": max(0, max_sector - config.allocation.max_sector_weight)
                    if constrained
                    else 0.0,
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
        pd.DataFrame(sector_rows),
    )
