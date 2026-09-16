import pandas as pd

from alex_quant.config import Config
from alex_quant.portfolio import allocate, rank_scores


def momentum_predictions(
    features: pd.DataFrame, predictions: pd.DataFrame, config: Config
) -> pd.DataFrame:
    """120-session price momentum, same signal dates and allocator as StockRanker."""
    rows = []
    for date, group in predictions.groupby("signal_date", sort=True):
        scores = features.xs(date, level="date")["return_120"].rename("score").reset_index()
        ranked = allocate(rank_scores(scores), config.universe, config.allocation)
        ranked["signal_date"] = date
        ranked["execution_date"] = group.execution_date.iloc[0]
        ranked["model_id"] = "momentum_120"
        rows.append(ranked)
    return pd.concat(rows, ignore_index=True)
