import numpy as np
import pandas as pd


def rank_scores(scores: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if scores.symbol.duplicated().any() or not np.isfinite(scores.score).all():
        raise ValueError("Las puntuaciones deben ser finitas y únicas por símbolo")
    if top_k < 1 or len(scores) < top_k:
        raise ValueError("No hay suficientes acciones para construir top_k posiciones")
    ranked = scores.sort_values(["score", "symbol"], ascending=[False, True]).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["selected"] = ranked["rank"] <= top_k
    ranked["target_weight"] = np.where(ranked.selected, 1 / top_k, 0.0)
    return ranked


def rebalance_after_costs(
    holdings: pd.Series,
    cash: float,
    weights: pd.Series,
    commission_rate: float,
    slippage_rate: float,
) -> tuple[pd.Series, float, float, float, pd.Series]:
    """Solve V_post = V_pre - rate * sum(abs(V_post * w - holdings))."""
    if (
        not weights.index.is_unique
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or weights.sum() > 1 + 1e-12
    ):
        raise ValueError("Solo se permiten carteras largas sin apalancamiento")
    if not weights.index.isin(holdings.index).all():
        raise ValueError("Hay pesos de símbolos sin precios disponibles")
    if cash < 0 or not np.isfinite(cash) or not np.isfinite(holdings).all() or (holdings < 0).any():
        raise ValueError("Tenencias o efectivo inválidos")
    pre = float(holdings.sum() + cash)
    rate = commission_rate + slippage_rate
    if pre <= 0 or not 0 <= rate < 1:
        raise ValueError("Capital o tasa de costes inválidos")
    weights = weights.reindex(holdings.index, fill_value=0.0)
    low, high = 0.0, pre
    for _ in range(60):
        post = (low + high) / 2
        cost = rate * float((weights * post - holdings).abs().sum())
        if post + cost > pre:
            high = post
        else:
            low = post
    desired = weights * ((low + high) / 2)
    trades = desired - holdings
    notional = float(trades.abs().sum())
    commission, slippage = notional * commission_rate, notional * slippage_rate
    remaining = pre - float(desired.sum()) - commission - slippage
    if remaining < -1e-7 * pre:
        raise ValueError("El rebalanceo produciría apalancamiento")
    return desired, max(0.0, remaining), commission, slippage, trades
