import os
import time
from datetime import date, datetime
from datetime import time as daytime
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from alex_quant.data.base import DataError, MarketDataProvider


class AlpacaProvider(MarketDataProvider):
    """Read-only Market Data REST client; never accesses the trading API."""

    URL = "https://data.alpaca.markets/v2/stocks/bars"

    def __init__(self, feed: str = "sip", transport: httpx.BaseTransport | None = None):
        self._key = os.getenv("ALPACA_API_KEY", "").strip()
        self._secret = os.getenv("ALPACA_SECRET_KEY", "").strip()
        if not self._key or not self._secret:
            raise DataError(
                "Faltan ALPACA_API_KEY y/o ALPACA_SECRET_KEY. Copie .env.example a .env "
                "y configure ambas variables con sus claves de Alpaca, o expórtelas en el entorno. "
                "Para una prueba explícita sin red: alex-quant run --config configs/fixture.toml "
                "--provider fixture. No se han sustituido los datos por datos sintéticos."
            )
        if feed not in {"sip", "iex"}:
            raise DataError("Feed de Alpaca inválido")
        self.feed = feed
        self._transport = transport

    @property
    def provenance(self) -> dict:
        return {"provider": "alpaca", "synthetic": False, "feed": self.feed, "adjustment": "all"}

    def fetch(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        if end >= datetime.now(ZoneInfo("America/New_York")).date():
            raise DataError("data.end debe ser anterior a hoy para excluir barras incompletas")
        market_timezone = ZoneInfo("America/New_York")
        params = {
            "symbols": ",".join(sorted(symbols)),
            "timeframe": "1Day",
            "start": datetime.combine(start, daytime.min, market_timezone).isoformat(),
            "end": datetime.combine(end, daytime(23, 59, 59), market_timezone).isoformat(),
            "adjustment": "all",
            "asof": str(end),
            "feed": self.feed,
            "sort": "asc",
            "limit": 10000,
        }
        headers = {"APCA-API-KEY-ID": self._key, "APCA-API-SECRET-KEY": self._secret}
        rows, seen_tokens = [], set()
        with httpx.Client(headers=headers, timeout=60, transport=self._transport) as client:
            while True:
                response = self._request(client, params)
                try:
                    payload = response.json()
                    for symbol, records in (payload.get("bars") or {}).items():
                        for bar in records:
                            session = (
                                pd.Timestamp(bar["t"])
                                .tz_convert("America/New_York")
                                .tz_localize(None)
                                .normalize()
                            )
                            rows.append(
                                {
                                    "date": session,
                                    "symbol": symbol,
                                    "open": bar["o"],
                                    "high": bar["h"],
                                    "low": bar["l"],
                                    "close": bar["c"],
                                    "volume": bar["v"],
                                }
                            )
                    token = payload.get("next_page_token")
                except (ValueError, KeyError, TypeError, AttributeError) as exc:
                    raise DataError("Respuesta OHLCV de Alpaca inválida") from exc
                if not token:
                    break
                if token in seen_tokens:
                    raise DataError("Alpaca repitió un token de paginación")
                seen_tokens.add(token)
                params["page_token"] = token
        if not rows:
            raise DataError(
                "Alpaca no devolvió barras. Revise fechas, símbolos y permisos del feed"
            )
        frame = pd.DataFrame(rows).set_index(["date", "symbol"]).sort_index()
        # Retain only the requested NY session labels.
        dates = frame.index.get_level_values("date")
        return frame.loc[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]

    @staticmethod
    def _request(client: httpx.Client, params: dict) -> httpx.Response:
        for attempt in range(3):
            try:
                response = client.get(AlpacaProvider.URL, params=params)
            except httpx.TransportError:
                if attempt == 2:
                    raise DataError("No se pudo conectar con Alpaca tras 3 intentos") from None
                time.sleep(2**attempt)
                continue
            if response.status_code == 200:
                return response
            if response.status_code in {401, 403}:
                raise DataError(
                    f"Alpaca HTTP {response.status_code}: revise sus claves y el permiso del "
                    "feed configurado (sip/iex). No se cambia de feed automáticamente."
                )
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < 2:
                    time.sleep(2**attempt)
                    continue
            # Never echo response bodies or headers, which might contain sensitive values.
            raise DataError(f"Alpaca HTTP {response.status_code}: no se pudo obtener el histórico")
        raise DataError("Alpaca no respondió")
