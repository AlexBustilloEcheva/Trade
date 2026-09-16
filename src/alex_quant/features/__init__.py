import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "return_5",
    "return_20",
    "return_60",
    "return_120",
    "volatility_20",
    "volatility_60",
    "distance_high_20",
    "distance_high_60",
    "drawdown",
    "relative_volume_20",
    "relative_spy_20",
    "relative_spy_60",
    "relative_sector_20",
    "relative_sector_60",
    "beta_spy_60",
]


def build_features(
    bars: pd.DataFrame, universe: dict[str, str], annualization: int = 252
) -> pd.DataFrame:
    """All features are known after close t; no centered windows, backfill or future fit."""
    closes = bars.close.unstack("symbol")
    volumes = bars.volume.unstack("symbol")
    daily = closes.pct_change(fill_method=None)
    spy_var = daily.SPY.rolling(60, min_periods=60).var(ddof=1)
    frames = []
    for symbol, sector in sorted(universe.items()):
        close, ret = closes[symbol], daily[symbol]
        features = pd.DataFrame(index=closes.index)
        for window in (5, 20, 60, 120):
            features[f"return_{window}"] = close.pct_change(window, fill_method=None)
        for window in (20, 60):
            features[f"volatility_{window}"] = ret.rolling(window, min_periods=window).std(
                ddof=1
            ) * np.sqrt(annualization)
            features[f"distance_high_{window}"] = (
                close / close.rolling(window, min_periods=window).max() - 1
            )
            for name, reference in (("spy", "SPY"), ("sector", sector)):
                features[f"relative_{name}_{window}"] = close.pct_change(
                    window, fill_method=None
                ) - closes[reference].pct_change(window, fill_method=None)
        features["drawdown"] = close / close.cummax() - 1
        # The denominator excludes today's volume; today's volume is known at the signal close.
        features["relative_volume_20"] = (
            volumes[symbol] / volumes[symbol].shift(1).rolling(20, min_periods=20).mean()
        )
        features["beta_spy_60"] = ret.rolling(60, min_periods=60).cov(daily.SPY) / spy_var.where(
            spy_var > 0
        )
        features["symbol"] = symbol
        frames.append(features.reset_index().set_index(["date", "symbol"])[FEATURE_COLUMNS])
    return pd.concat(frames).sort_index().replace([np.inf, -np.inf], np.nan)
