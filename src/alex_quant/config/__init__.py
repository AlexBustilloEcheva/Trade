"""Validated, file-based research configuration."""

import json
import re
import tomllib
from datetime import date
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class DataConfig(StrictModel):
    start: date
    end: date
    feed: Literal["sip", "iex"] = "sip"
    adjustment: Literal["all"] = "all"
    calendar: Literal["XNYS"] = "XNYS"
    benchmarks: list[str] = Field(default_factory=lambda: ["SPY", "QQQ"])

    @model_validator(mode="after")
    def validate_dates(self) -> Self:
        if self.start >= self.end:
            raise ValueError("data.start debe ser anterior a data.end")
        if "SPY" not in self.benchmarks or len(set(self.benchmarks)) != len(self.benchmarks):
            raise ValueError("benchmarks debe incluir SPY y no tener duplicados")
        return self


class ResearchConfig(StrictModel):
    label_mode: Literal["close_to_close", "rebalance_open_to_open"] = "close_to_close"
    horizon_sessions: int = Field(default=10, ge=1)
    rebalance_frequency: Literal["W-FRI", "W-MON", "daily", "monthly"] = "W-FRI"
    minimum_history_sessions: int = Field(default=120, ge=120)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


class WalkForwardConfig(StrictModel):
    train_sessions: int = Field(default=504, ge=2)
    min_train_sessions: int = Field(default=252, ge=2)
    retrain_every_signals: int = Field(default=4, ge=1)
    embargo_sessions: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if self.min_train_sessions > self.train_sessions:
            raise ValueError("min_train_sessions no puede superar train_sessions")
        return self


class PortfolioConfig(StrictModel):
    initial_capital: float = Field(default=100000.0, gt=0)
    commission_bps: float = Field(default=5.0, ge=0, lt=1000)
    slippage_bps: float = Field(default=5.0, ge=0, lt=1000)
    cash_annual_rate: float = Field(default=0.0, ge=0, le=1)
    annualization: int = Field(default=252, ge=1)


class AllocationConfig(StrictModel):
    top_k: int = Field(default=5, ge=1)
    max_asset_weight: float = Field(default=0.20, gt=0, le=1)
    max_sector_weight: float = Field(default=0.40, gt=0, le=1)


class ModelConfig(StrictModel):
    n_estimators: int = Field(default=120, ge=1)
    learning_rate: float = Field(default=0.03, gt=0, le=1)
    num_leaves: int = Field(default=15, ge=2)
    min_child_samples: int = Field(default=40, ge=1)
    reg_lambda: float = Field(default=1.0, ge=0)


class Config(StrictModel):
    data: DataConfig
    universe: dict[str, str]
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    allocation: AllocationConfig = Field(default_factory=AllocationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_top_k(cls, value: dict) -> dict:
        value = dict(value)
        research = dict(value.get("research", {}))
        if "top_k" in research:
            allocation = dict(value.get("allocation", {}))
            if "top_k" in allocation:
                raise ValueError("Use solo allocation.top_k; research.top_k es una clave antigua")
            allocation["top_k"] = research.pop("top_k")
            value.update(research=research, allocation=allocation)
        return value

    @model_validator(mode="after")
    def validate_universe(self) -> Self:
        if not self.universe:
            raise ValueError("El universo no puede estar vacío")
        if set(self.universe) & (set(self.universe.values()) | set(self.data.benchmarks)):
            raise ValueError("Las acciones y los ETF deben ser símbolos distintos")
        if any(not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,14}", s) for s in self.symbols):
            raise ValueError("Símbolo inválido; utilice tickers estadounidenses en mayúsculas")
        return self

    @property
    def symbols(self) -> list[str]:
        return sorted(set(self.universe) | set(self.universe.values()) | set(self.data.benchmarks))


def _read(path: Path, ancestors: frozenset[Path] = frozenset()) -> dict:
    path = path.resolve()
    if path in ancestors:
        raise ValueError("Referencia circular entre archivos de configuración")
    with path.open("rb") as stream:
        child = tomllib.load(stream)
    parent = child.pop("extends", None)
    if parent is None:
        return child
    base = _read(path.parent / parent, ancestors | {path})
    for key, value in child.items():
        # Universe overrides replace the whole universe, rather than accidentally adding stocks.
        if key != "universe" and isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key].update(value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if path.suffix == ".json":
        return Config.model_validate(json.loads(path.read_text(encoding="utf-8")))
    return Config.model_validate(_read(path))
