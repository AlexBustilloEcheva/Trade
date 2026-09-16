from dataclasses import dataclass

import numpy as np
import pandas as pd

from alex_quant.config import Config
from alex_quant.data.base import DataError
from alex_quant.data.schedule import rebalance_schedule, signal_dates  # noqa: F401
from alex_quant.features import FEATURE_COLUMNS
from alex_quant.models import StockRanker
from alex_quant.portfolio import allocate, rank_scores


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    training_windows: list[dict]
    models: dict[str, str]


def walk_forward(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    config: Config,
    schedule: pd.DataFrame | None = None,
) -> WalkForwardResult:
    if not features.index.is_unique or not features.index.is_monotonic_increasing:
        raise DataError("Variables duplicadas o desordenadas")
    if not features.index.equals(labels.index):
        raise DataError("Índices de variables y etiquetas desalineados")
    data = features.join(labels)
    eligible = data.dropna(subset=FEATURE_COLUMNS)
    earliest = sessions[min(config.research.minimum_history_sessions, len(sessions) - 1)]
    eligible = eligible.loc[eligible.index.get_level_values("date") >= earliest]
    predictions, windows, models = [], [], {}
    model, since_fit = None, 0
    if schedule is None:
        schedule = rebalance_schedule(sessions, config.research.rebalance_frequency)
    for signal in schedule.index:
        dates = eligible.index.get_level_values("date")
        current = eligible.loc[dates == signal]
        if len(current) != len(config.universe):
            if model is not None:
                raise DataError("Faltan variables para una señal programada durante la evaluación")
            continue
        if model is None or since_fit >= config.walk_forward.retrain_every_signals:
            position = sessions.get_loc(signal) - config.walk_forward.embargo_sessions
            if position < 1:
                continue
            cutoff = sessions[position]
            train = eligible.loc[(dates < signal) & (eligible.label_end < cutoff)].dropna(
                subset=["target", "label_end"]
            )
            train_dates = train.index.get_level_values("date").unique()
            train_dates = train_dates[-config.walk_forward.train_sessions :]
            if len(train_dates) < config.walk_forward.min_train_sessions:
                continue
            train = train.loc[train.index.get_level_values("date").isin(train_dates)]
            if train.label_end.max() >= cutoff or train_dates.max() >= signal:
                raise DataError("Solapamiento temporal detectado en entrenamiento")
            model = StockRanker(config)
            model.fit(train[FEATURE_COLUMNS], train.target)
            model_id = f"model_{len(windows):04d}"
            models[model_id] = model.export_model()
            windows.append(
                {
                    "model_id": model_id,
                    "train_start": str(train_dates.min().date()),
                    "train_end": str(train_dates.max().date()),
                    "train_dates": [str(day.date()) for day in train_dates],
                    "label_end_max": str(train.label_end.max().date()),
                    "label_cutoff_exclusive": str(cutoff.date()),
                    "prediction_start": str(signal.date()),
                    "prediction_end": str(signal.date()),
                    "n_train_sessions": len(train_dates),
                    "n_train_rows": len(train),
                    "seed": config.research.seed,
                }
            )
            since_fit = 0
        scores = model.predict(current[FEATURE_COLUMNS])
        if not np.isfinite(scores).all():
            raise DataError("El modelo generó puntuaciones no finitas")
        ranked = rank_scores(
            pd.DataFrame(
                {
                    "symbol": current.index.get_level_values("symbol"),
                    "score": scores,
                }
            ),
        )
        ranked = allocate(ranked, config.universe, config.allocation)
        ranked["signal_date"] = signal
        ranked["execution_date"] = schedule.loc[signal, "entry_date"]
        ranked["model_id"] = windows[-1]["model_id"]
        predictions.append(ranked)
        windows[-1]["prediction_end"] = str(signal.date())
        since_fit += 1
    if not predictions:
        raise DataError(
            "Histórico insuficiente para predecir: se necesitan el calentamiento de variables, "
            "min_train_sessions, horizonte de etiquetas y una siguiente sesión de ejecución"
        )
    return WalkForwardResult(pd.concat(predictions, ignore_index=True), windows, models)
