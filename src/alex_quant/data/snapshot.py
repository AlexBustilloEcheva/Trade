import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd

from alex_quant.data.base import DataError, MarketDataProvider


class SnapshotProvider(MarketDataProvider):
    """Replay an immutable local snapshot, preserving its original source designation."""

    def __init__(self, directory: Path):
        self.path = directory / "input_bars.csv"
        try:
            self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            expected = self.manifest["files"]["input_bars.csv"]
            if hashlib.sha256(self.path.read_bytes()).hexdigest() != expected:
                raise DataError("El hash del snapshot no coincide; los datos han cambiado")
            if not isinstance(self.manifest["source"]["synthetic"], bool):
                raise DataError("El snapshot no declara si sus datos son sintéticos")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise DataError(f"Snapshot inválido: {exc}") from exc

    @property
    def provenance(self) -> dict:
        return dict(self.manifest["source"])

    def fetch(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        frame = pd.read_csv(self.path, parse_dates=["date"], float_precision="round_trip")
        # Do not sort: validation must detect reordered or corrupted snapshots.
        return frame.set_index(["date", "symbol"])
