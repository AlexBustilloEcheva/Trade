from datetime import date

from alex_quant.data.base import DataError, MarketDataProvider, trading_sessions, validate_bars


def diagnostic_sessions(start: date, end: date):
    if not 1 <= (end - start).days <= 10:
        raise DataError("El diagnóstico requiere un intervalo de 2 a 11 días naturales")
    sessions = trading_sessions(start, end)
    if not 2 <= len(sessions) <= 5:
        raise DataError("El diagnóstico requiere entre 2 y 5 sesiones históricas de SPY")
    return sessions


def diagnose_data(provider: MarketDataProvider, start: date, end: date) -> dict:
    sessions = diagnostic_sessions(start, end)
    bars = validate_bars(provider.fetch(["SPY"], start, end), ["SPY"], sessions)
    return {
        "status": "ok",
        "symbol": "SPY",
        "feed": provider.provenance.get("feed"),
        "bars": len(bars),
        "start": str(sessions[0].date()),
        "end": str(sessions[-1].date()),
        "ohlcv_valid": True,
    }
