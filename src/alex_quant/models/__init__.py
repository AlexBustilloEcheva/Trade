import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from alex_quant.config import Config


class StockRanker:
    """One global LightGBM regressor; scores rank stocks within each signal date."""

    def __init__(self, config: Config):
        self.model = LGBMRegressor(
            objective="regression",
            **config.model.model_dump(),
            random_state=config.research.seed,
            deterministic=True,
            force_col_wise=True,
            n_jobs=1,
            verbosity=-1,
        )

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        self.model.fit(features, target)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.model.predict(features)

    def export_model(self) -> str:
        return self.model.booster_.model_to_string()
