from pathlib import Path

from alex_quant.backtest.engine import simulate
from alex_quant.backtest.walk_forward import walk_forward
from alex_quant.config import Config
from alex_quant.data.base import MarketDataProvider, trading_sessions, validate_bars
from alex_quant.evaluation import calculate_metrics
from alex_quant.features import build_features
from alex_quant.labels import build_labels
from alex_quant.reports import write_artifacts


def run_pipeline(config: Config, provider: MarketDataProvider, output: Path) -> dict:
    sessions = trading_sessions(config.data.start, config.data.end)
    bars = validate_bars(
        provider.fetch(config.symbols, config.data.start, config.data.end), config.symbols, sessions
    )
    features = build_features(bars, config.universe, config.portfolio.annualization)
    labels = build_labels(bars, config.universe, config.research.horizon_sessions)
    walk = walk_forward(features, labels, sessions, config)
    backtest = simulate(bars, walk.predictions, config)
    metrics = {
        name: calculate_metrics(
            group, config.portfolio.annualization, config.portfolio.cash_annual_rate
        )
        for name, group in backtest.equity.groupby("portfolio", sort=True)
    }
    write_artifacts(output, config, bars, provider.provenance, walk, backtest, metrics)
    return metrics
