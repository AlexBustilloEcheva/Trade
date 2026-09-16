from pathlib import Path

import pandas as pd

from alex_quant.backtest.benchmarks import momentum_predictions
from alex_quant.backtest.engine import simulate
from alex_quant.backtest.walk_forward import walk_forward
from alex_quant.config import Config
from alex_quant.data.base import MarketDataProvider, trading_sessions, validate_bars
from alex_quant.data.schedule import rebalance_schedule
from alex_quant.evaluation import calculate_metrics
from alex_quant.evaluation.research import (
    exposure_metrics,
    rank_information_coefficient,
    yearly_metrics,
)
from alex_quant.features import build_features
from alex_quant.labels import build_labels
from alex_quant.reports import write_artifacts


def run_pipeline(config: Config, provider: MarketDataProvider, output: Path) -> dict:
    sessions = trading_sessions(config.data.start, config.data.end)
    bars = validate_bars(
        provider.fetch(config.symbols, config.data.start, config.data.end), config.symbols, sessions
    )
    features = build_features(bars, config.universe, config.portfolio.annualization)
    schedule = rebalance_schedule(sessions, config.research.rebalance_frequency)
    labels = build_labels(
        bars,
        config.universe,
        config.research.horizon_sessions,
        mode=config.research.label_mode,
        schedule=schedule,
    )
    walk = walk_forward(features, labels, sessions, config, schedule)
    momentum = momentum_predictions(features, walk.predictions, config)
    backtest = simulate(bars, walk.predictions, config, momentum, schedule)
    stress = simulate(bars, walk.predictions, config, momentum, schedule, cost_multiplier=2.0)
    metrics = {
        name: calculate_metrics(
            group, config.portfolio.annualization, config.portfolio.cash_annual_rate
        )
        | exposure_metrics(group)
        for name, group in backtest.equity.groupby("portfolio", sort=True)
    }
    stress_metrics = {
        name: calculate_metrics(
            group, config.portfolio.annualization, config.portfolio.cash_annual_rate
        )
        for name, group in stress.equity.groupby("portfolio", sort=True)
    }
    information_coefficient = pd.concat(
        [
            rank_information_coefficient(frame, labels, sessions[-1]).assign(portfolio=name)
            for name, frame in (("stockranker", walk.predictions), ("momentum", momentum))
        ],
        ignore_index=True,
    )
    diagnostics = {
        "yearly": yearly_metrics(backtest.equity),
        "information_coefficient": information_coefficient,
        "momentum_predictions": momentum,
        "labels": labels.reset_index(),
        "schedule": schedule.reset_index(),
        "equity_stress": stress.equity,
        "trades_stress": stress.trades,
        "positions_stress": stress.positions,
    }
    write_artifacts(
        output,
        config,
        bars,
        provider.provenance,
        walk,
        backtest,
        metrics,
        diagnostics,
        stress_metrics,
    )
    return metrics
