"""Small deterministic synthetic fixture generator, available only by explicit selection."""

import hashlib
from datetime import date

import numpy as np
import pandas as pd

from alex_quant.data.base import MarketDataProvider, trading_sessions


class FixtureProvider(MarketDataProvider):
    def __init__(self, seed: int = 42):
        self.seed = seed

    @property
    def provenance(self) -> dict:
        return {"provider": "fixture", "synthetic": True, "seed": self.seed, "version": 1}

    def fetch(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        sessions = trading_sessions(start, end)
        t = np.arange(len(sessions), dtype=float)
        frames = []
        for symbol in sorted(symbols):
            stable_id = int.from_bytes(hashlib.sha256(symbol.encode()).digest()[:4], "little")
            rng = np.random.default_rng([self.seed, stable_id])
            phase = stable_id % 31 / 5
            returns = 0.0003 + 0.003 * np.sin(t / 17 + phase) + rng.normal(0, 0.008, len(t))
            close = (60 + stable_id % 120) * np.exp(np.cumsum(returns))
            previous = np.r_[close[0] / np.exp(returns[0]), close[:-1]]
            opening = previous * np.exp(returns * 0.35)
            frames.append(
                pd.DataFrame(
                    {
                        "date": sessions,
                        "symbol": symbol,
                        "open": opening,
                        "high": np.maximum(opening, close) * 1.003,
                        "low": np.minimum(opening, close) * 0.997,
                        "close": close,
                        "volume": 1000000 + 100000 * np.sin(t / 7 + phase),
                    }
                )
            )
        return pd.concat(frames).set_index(["date", "symbol"]).sort_index()
