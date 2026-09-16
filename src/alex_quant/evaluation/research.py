import numpy as np
import pandas as pd


def yearly_metrics(equity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, curve in equity.groupby("portfolio", sort=True):
        curve = curve.sort_values("date")
        observed = curve.loc[~curve.is_baseline].copy()
        for year, group in observed.groupby(observed.date.dt.year):
            # Reset yearly NAV to 1 before the first return, including any January loss.
            nav = np.r_[1.0, (1 + group.daily_return).cumprod().to_numpy()]
            rows.append(
                {
                    "portfolio": name,
                    "year": int(year),
                    "sessions": len(group),
                    "return": float(nav[-1] - 1),
                    "max_drawdown": float((nav / np.maximum.accumulate(nav) - 1).min()),
                }
            )
    return pd.DataFrame(rows)


def rank_information_coefficient(
    predictions: pd.DataFrame, labels: pd.DataFrame, asof: pd.Timestamp
) -> pd.DataFrame:
    joined = predictions.merge(
        labels.reset_index().rename(columns={"date": "signal_date"}),
        on=["signal_date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    rows = []
    for date, group in joined.groupby("signal_date", sort=True):
        mature = group.loc[(group.label_end <= asof) & group.target.notna()]
        # Require the whole cross section; never silently evaluate a changing subset.
        if len(mature) != len(group):
            continue
        scores, targets = mature.score.rank(method="average"), mature.target.rank(method="average")
        ic = (
            float(scores.corr(targets))
            if len(mature) >= 2 and scores.nunique() > 1 and targets.nunique() > 1
            else np.nan
        )
        rows.append(
            {
                "signal_date": date,
                "label_end": mature.label_end.max(),
                "n_assets": len(mature),
                "spearman_ic": ic,
            }
        )
    return pd.DataFrame(rows, columns=["signal_date", "label_end", "n_assets", "spearman_ic"])


def exposure_metrics(curve: pd.DataFrame) -> dict:
    observed = curve.loc[~curve.is_baseline]
    return {
        "average_cash_weight": float(observed.cash_weight.mean()),
        "average_max_sector_weight": float(observed.max_sector_weight.mean()),
        "peak_asset_weight": float(observed.max_asset_weight.max()),
        "peak_sector_weight": float(observed.max_sector_weight.max()),
        "asset_limit_breach_sessions": int((observed.asset_limit_excess > 1e-12).sum()),
        "sector_limit_breach_sessions": int((observed.sector_limit_excess > 1e-12).sum()),
        "max_asset_limit_excess": float(observed.asset_limit_excess.max()),
        "max_sector_limit_excess": float(observed.sector_limit_excess.max()),
        "limits_applicable": bool(observed.limits_applicable.all()),
    }
