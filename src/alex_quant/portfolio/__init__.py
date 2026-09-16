import numpy as np
import pandas as pd

from alex_quant.config import AllocationConfig


def rank_scores(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.symbol.duplicated().any() or not np.isfinite(scores.score).all():
        raise ValueError("Las puntuaciones deben ser finitas y únicas por símbolo")
    ranked = scores.sort_values(["score", "symbol"], ascending=[False, True]).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def allocate(
    ranked: pd.DataFrame, universe: dict[str, str], config: AllocationConfig
) -> pd.DataFrame:
    """Greedy fixed-slot allocation in ranking order, skipping saturated sectors."""
    result = ranked.copy()
    result["sector_etf"] = result.symbol.map(universe)
    if result.sector_etf.isna().any():
        raise ValueError("Falta el sector de un candidato")
    weights, sector_weights = [], {}
    count, invested = 0, 0.0
    slot = min(1 / config.top_k, config.max_asset_weight)
    for row in result.itertuples():
        capacity = config.max_sector_weight - sector_weights.get(row.sector_etf, 0.0)
        weight = min(slot, capacity, 1 - invested) if count < config.top_k else 0.0
        weight = max(0.0, weight) if weight > 1e-12 else 0.0
        weights.append(weight)
        if weight:
            count += 1
            invested += weight
            sector_weights[row.sector_etf] = sector_weights.get(row.sector_etf, 0.0) + weight
    result["target_weight"] = weights
    result["selected"] = result.target_weight > 0
    return result


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
