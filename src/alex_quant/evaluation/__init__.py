import numpy as np
import pandas as pd


def calculate_metrics(
    curve: pd.DataFrame, annualization: int = 252, risk_free_annual: float = 0.0
) -> dict:
    """First row is initial capital, not a return observation. Undefined ratios become null."""
    if len(curve) < 2 or not curve.date.is_monotonic_increasing or curve.date.duplicated().any():
        raise ValueError("La curva debe contener una base y sesiones ordenadas sin duplicados")
    if not np.isfinite(curve.equity).all() or (curve.equity <= 0).any():
        raise ValueError("Patrimonio inválido")
    returns = curve.equity.pct_change(fill_method=None).iloc[1:]
    periods = len(returns)
    cumulative = float(curve.equity.iloc[-1] / curve.equity.iloc[0] - 1)
    annual = float((1 + cumulative) ** (annualization / periods) - 1)
    rf_daily = (1 + risk_free_annual) ** (1 / annualization) - 1
    excess = returns - rf_daily
    std = float(returns.std(ddof=1)) if periods > 1 else 0.0
    downside = float(np.sqrt(np.mean(np.minimum(excess.to_numpy(), 0) ** 2)))
    drawdown = curve.equity / curve.equity.cummax() - 1
    max_dd = float(drawdown.min())
    observations = curve.iloc[1:]
    cost = float(observations.cost.sum())
    return {
        "cumulative_return": cumulative,
        "annualized_return": annual,
        "annualized_volatility": std * np.sqrt(annualization),
        "sharpe": float(excess.mean() / std * np.sqrt(annualization)) if std > 1e-15 else None,
        "sortino": (
            float(excess.mean() / downside * np.sqrt(annualization)) if downside > 1e-15 else None
        ),
        "max_drawdown": max_dd,
        "calmar": annual / abs(max_dd) if max_dd < -1e-15 else None,
        "turnover_total": float(observations.turnover.sum()),
        "turnover_annualized": float(observations.turnover.sum()) * annualization / periods,
        "costs_total": cost,
        "costs_fraction_initial": cost / float(curve.equity.iloc[0]),
        "commissions_total": float(observations.commission.sum()),
        "slippage_total": float(observations.slippage.sum()),
        "rebalance_count": int(observations.rebalanced.sum()),
        "average_exposure": float(observations.exposure.mean()),
        "sessions": periods,
        "initial_equity": float(curve.equity.iloc[0]),
        "final_equity": float(curve.equity.iloc[-1]),
        "start": str(pd.Timestamp(curve.date.iloc[0]).date()),
        "end": str(pd.Timestamp(curve.date.iloc[-1]).date()),
    }
