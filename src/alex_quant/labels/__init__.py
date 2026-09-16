import pandas as pd


def build_labels(bars: pd.DataFrame, universe: dict[str, str], horizon: int) -> pd.DataFrame:
    """y(t) = C_stock(t+h)/C_stock(t) - C_sector(t+h)/C_sector(t)."""
    if horizon < 1:
        raise ValueError("El horizonte debe ser positivo")
    closes = bars.close.unstack("symbol")
    future = closes.shift(-horizon) / closes - 1
    end_dates = pd.Series(closes.index, index=closes.index).shift(-horizon)
    frames = []
    for symbol, sector in sorted(universe.items()):
        frame = pd.DataFrame(
            {
                "target": future[symbol] - future[sector],
                "label_end": end_dates,
                "symbol": symbol,
            }
        )
        frames.append(frame.reset_index().set_index(["date", "symbol"]))
    return pd.concat(frames).sort_index()
