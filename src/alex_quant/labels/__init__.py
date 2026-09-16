import pandas as pd


def build_labels(
    bars: pd.DataFrame,
    universe: dict[str, str],
    horizon: int = 10,
    *,
    mode: str = "close_to_close",
    schedule: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Relative returns on precisely matched entry/exit observations for stock and ETF."""
    if horizon < 1:
        raise ValueError("El horizonte debe ser positivo")
    closes = bars.close.unstack("symbol")
    future = closes.shift(-horizon) / closes - 1
    end_dates = pd.Series(closes.index, index=closes.index).shift(-horizon)
    entry_dates = pd.Series(closes.index, index=closes.index)
    if mode == "rebalance_open_to_open":
        if schedule is None:
            raise ValueError("El objetivo negociable requiere un calendario compartido")
        opens = bars.open.unstack("symbol")
        timing = schedule.reindex(closes.index)
        entry_dates, end_dates = timing.entry_date, timing.label_end
        entry_prices = opens.reindex(pd.DatetimeIndex(entry_dates)).set_axis(closes.index)
        exit_prices = opens.reindex(pd.DatetimeIndex(end_dates)).set_axis(closes.index)
        future = exit_prices / entry_prices - 1
    elif mode != "close_to_close":
        raise ValueError("Modo de etiqueta desconocido")
    frames = []
    for symbol, sector in sorted(universe.items()):
        frame = pd.DataFrame(
            {
                "target": future[symbol] - future[sector],
                "entry_date": entry_dates,
                "label_end": end_dates,
                "symbol": symbol,
            }
        )
        frames.append(frame.reset_index().set_index(["date", "symbol"]))
    return pd.concat(frames).sort_index()
