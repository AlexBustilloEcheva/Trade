"""A shared, observed-session schedule. Never extends the requested price interval."""

import pandas as pd


def rebalance_schedule(sessions: pd.DatetimeIndex, frequency: str) -> pd.DataFrame:
    if not sessions.is_unique or not sessions.is_monotonic_increasing:
        raise ValueError("Calendario duplicado o desordenado")
    if frequency not in {"daily", "monthly", "W-FRI", "W-MON"}:
        raise ValueError("Frecuencia de señales inválida")
    if frequency == "daily":
        signals, entries = sessions[:-1], sessions[1:]
    else:
        periods = sessions.to_period("M" if frequency == "monthly" else frequency)
        # A boundary exists only when the next observed session is in another period.
        boundary = periods[:-1] != periods[1:]
        signals, entries = sessions[:-1][boundary], sessions[1:][boundary]
    schedule = pd.DataFrame(
        {"entry_date": entries}, index=pd.DatetimeIndex(signals, name="signal_date")
    )
    schedule["label_end"] = schedule.entry_date.shift(-1)
    return schedule


def signal_dates(sessions: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    return rebalance_schedule(sessions, frequency).index.rename("date")
