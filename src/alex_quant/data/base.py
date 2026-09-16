from abc import ABC, abstractmethod
from datetime import date, timedelta

import exchange_calendars as xcals
import numpy as np
import pandas as pd


class DataError(ValueError):
    """Missing, inconsistent or inaccessible market data."""


class MarketDataProvider(ABC):
    @abstractmethod
    def fetch(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        """Return sorted (date, symbol) OHLCV, dates being naive NY session labels."""

    @property
    @abstractmethod
    def provenance(self) -> dict:
        """Non-secret source metadata, including whether prices are synthetic."""


def trading_sessions(start: date, end: date) -> pd.DatetimeIndex:
    calendar = xcals.get_calendar(
        "XNYS", start=str(start - timedelta(days=7)), end=str(end + timedelta(days=7))
    )
    return calendar.sessions_in_range(str(start), str(end)).tz_localize(None).rename("date")


def validate_bars(
    bars: pd.DataFrame, symbols: list[str], sessions: pd.DatetimeIndex
) -> pd.DataFrame:
    if not isinstance(bars.index, pd.MultiIndex) or bars.index.names != ["date", "symbol"]:
        raise DataError("Se requiere un índice (date, symbol)")
    if bars.empty or len(sessions) < 2:
        raise DataError("No hay suficientes sesiones de datos")
    if bars.index.has_duplicates:
        raise DataError("Índices duplicados en los datos de mercado")
    if not bars.index.is_monotonic_increasing:
        raise DataError("Los datos no están en orden temporal (date, symbol)")
    dates = bars.index.get_level_values("date")
    if not isinstance(dates, pd.DatetimeIndex) or dates.tz is not None:
        raise DataError("Las fechas deben ser sesiones locales sin zona horaria")
    expected = pd.MultiIndex.from_product([sessions, sorted(symbols)], names=["date", "symbol"])
    missing, extra = expected.difference(bars.index), bars.index.difference(expected)
    if len(missing) or len(extra):
        raise DataError(
            f"Datos ausentes o fuera del calendario: faltan {len(missing)}, sobran {len(extra)}. "
            f"Ejemplos ausentes: {list(missing[:3])}. No se rellenan precios automáticamente."
        )
    columns = ["open", "high", "low", "close", "volume"]
    if not set(columns).issubset(bars.columns):
        raise DataError(f"Faltan columnas OHLCV: {columns}")
    try:
        values = bars[columns].astype(float)
    except (ValueError, TypeError) as exc:
        raise DataError("OHLCV debe ser numérico") from exc
    if not np.isfinite(values.to_numpy()).all():
        raise DataError("Datos ausentes o no finitos en OHLCV")
    if (values[["open", "high", "low", "close"]] <= 0).any().any():
        raise DataError("Los precios deben ser positivos")
    if (values.volume <= 0).any():
        raise DataError("Volumen cero o negativo: revise la cobertura del proveedor")
    if (values.high < values[["open", "close", "low"]].max(axis=1)).any() or (
        values.low > values[["open", "close", "high"]].min(axis=1)
    ).any():
        raise DataError("OHLC inconsistente: máximos o mínimos inválidos")
    return values
