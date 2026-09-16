import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from alex_quant.backtest.engine import BacktestResult
from alex_quant.backtest.walk_forward import WalkForwardResult
from alex_quant.config import Config


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _equity_svg(equity: pd.DataFrame) -> str:
    colors = {
        "stockranker": "#2563eb",
        "SPY": "#dc2626",
        "equal_weight": "#059669",
        "cash": "#64748b",
    }
    pivot = equity.pivot(index="date", columns="portfolio", values="equity")
    pivot = pivot / pivot.iloc[0] * 100
    low, high = float(pivot.min().min()), float(pivot.max().max())
    span = max(high - low, 1.0)
    low, high = low - span * 0.05, high + span * 0.05
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 420" role="img" '
        'aria-label="Equity normalizada, base 100">',
        '<rect width="900" height="420" fill="white"/>',
        '<g font-family="sans-serif" font-size="13">',
        '<text x="65" y="24">Alex Quant Lab · Equity neta (base 100)</text>',
    ]
    for step in range(5):
        value = low + (high - low) * step / 4
        y = 350 - 290 * step / 4
        parts.append(f'<path d="M65 {y:.2f} H865" stroke="#e2e8f0"/>')
        parts.append(f'<text x="8" y="{y + 4:.2f}">{value:.1f}</text>')
    for i, (name, color) in enumerate(colors.items()):
        points = " ".join(
            f"{65 + 800 * j / max(len(pivot) - 1, 1):.2f},"
            f"{350 - 290 * (v - low) / (high - low):.2f}"
            for j, v in enumerate(pivot[name])
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{65 + i * 200}" y="400" fill="{color}">{name}</text>')
    parts.extend(
        [
            f'<text x="65" y="375">{pivot.index[0].date()}</text>',
            f'<text x="785" y="375">{pivot.index[-1].date()}</text>',
            "</g></svg>",
        ]
    )
    return "\n".join(parts)


def _report(metrics: dict, config: Config, source: dict, data_hash: str, n_models: int) -> str:
    designation = (
        "**DATOS SINTÉTICOS DE PRUEBA. Estas cifras no son resultados de mercado.**"
        if source["synthetic"]
        else "Datos históricos de Alpaca; simulación de investigación."
    )
    rows = []
    for name, m in metrics.items():
        sharpe = "n/d" if m["sharpe"] is None else f"{m['sharpe']:.3f}"
        rows.append(
            f"| {name} | {m['cumulative_return']:.2%} | {m['annualized_return']:.2%} | "
            f"{m['annualized_volatility']:.2%} | {sharpe} | {m['max_drawdown']:.2%} | "
            f"{m['costs_total']:.2f} | {m['rebalance_count']} | {m['average_exposure']:.2%} |"
        )
    first = metrics["stockranker"]
    table_header = (
        "| Cartera | Acumulada | Anualizada | Volatilidad | Sharpe | Máx. drawdown | "
        "Costes USD | Rebalanceos | Exposición |"
    )
    replay_command = (
        "uv run --frozen alex-quant run --config RUTA/config.json --provider snapshot "
        "--snapshot RUTA --output artifacts/replay"
    )
    return f"""# Alex Quant Lab — StockRanker

{designation}

Periodo común: {first["start"]} a {first["end"]} ({first["sessions"]} sesiones de rentabilidad).
Capital inicial por cartera: {config.portfolio.initial_capital:,.2f} USD.

{table_header}
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

![Equity neta](equity.svg)

Sortino, Calmar, rotación, comisiones y deslizamiento separados: [metrics.json](metrics.json).
Los cocientes indefinidos se exportan como `null`, sin infinitos ni NaN.

## Método

- Modelo LightGBM global de regresión; ranking de retorno a {config.research.horizon_sessions}
  sesiones menos retorno del ETF sectorial. Selección de {config.research.top_k} acciones.
- Variables al cierre; ejecución en la siguiente apertura; nunca en el cierre de la señal.
- Frecuencia: {config.research.rebalance_frequency}. Equiponderación tras descontar costes;
  pesos variables entre rebalanceos. Fracciones de acción y exposición larga hasta 100%.
- {n_models} modelos; hasta {config.walk_forward.train_sessions} sesiones de entrenamiento,
  mínimo {config.walk_forward.min_train_sessions}; etiquetas terminadas antes del corte.
  Véanse [training_windows.json](training_windows.json) y [models.json](models.json).
- Costes por nominal comprado y vendido: {config.portfolio.commission_bps} pb de comisión
  y {config.portfolio.slippage_bps} pb de deslizamiento, también en benchmarks invertidos.
- SPY se compra una vez; equal_weight rebalancea todo el universo en las mismas fechas.
  Efectivo y tipo libre de riesgo: {config.portfolio.cash_annual_rate:.2%} anual efectivo.
- Se conserva la cartera al final, sin liquidación ni coste terminal ficticio.

## Reproducción

Snapshot SHA-256: `{data_hash}`. Procedencia: `{json.dumps(source, sort_keys=True)}`.
Configuración resuelta, versiones, hashes del código y de archivos: `config.json` y `manifest.json`.
Desde el repositorio, con las dependencias de `uv.lock`, sustituya RUTA por este directorio:

```shell
{replay_command}
```

El snapshot evita que una nueva descarga revisada cambie los resultados. Los modelos se
reentrenan con las mismas semillas; comparar requiere también el mismo código y entorno.

## Limitaciones

Universo y sectores fijos: sesgo de supervivencia y clasificaciones históricas aproximadas.
Alpaca entrega barras ajustadas actuales, no un archivo de publicaciones point-in-time;
puede revisar precios y acciones corporativas. Todas las variables son cocientes o
retornos causales, pero la calidad histórica del proveedor no queda garantizada.
Los ajustes representan una aproximación a retorno total, no una contabilidad de
dividendos y escisiones con sus fechas de pago. IEX cubre una sola bolsa si se selecciona.
No se modelan subastas, capacidad, impacto variable, impuestos ni fallos de ejecución.
Las etiquetas de diez sesiones se solapan; no hay tuning ni inferencia estadística.
No hay órdenes reales, paper trading, cortos ni apalancamiento.
"""


def write_artifacts(
    output: Path,
    config: Config,
    bars: pd.DataFrame,
    source: dict,
    walk: WalkForwardResult,
    backtest: BacktestResult,
    metrics: dict,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".alex-quant-", dir=output.parent) as temporary:
        directory = Path(temporary)
        bars.to_csv(directory / "input_bars.csv", float_format="%.17g", lineterminator="\n")
        for name, frame in (
            ("equity", backtest.equity),
            ("predictions", walk.predictions),
            ("positions", backtest.positions),
            ("trades", backtest.trades),
        ):
            frame.to_csv(
                directory / f"{name}.csv", index=False, float_format="%.17g", lineterminator="\n"
            )
        write_json(directory / "config.json", config.model_dump(mode="json"))
        write_json(directory / "metrics.json", metrics)
        write_json(directory / "training_windows.json", walk.training_windows)
        write_json(directory / "models.json", walk.models)
        data_hash = hashlib.sha256((directory / "input_bars.csv").read_bytes()).hexdigest()
        (directory / "report.md").write_text(
            _report(metrics, config, source, data_hash, len(walk.models)), encoding="utf-8"
        )
        (directory / "equity.svg").write_text(_equity_svg(backtest.equity), encoding="utf-8")
        package_root = Path(__file__).resolve().parents[1]
        source_hashes = {
            str(path.relative_to(package_root)).replace("\\", "/"): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(package_root.rglob("*.py"))
        }
        dependencies = {
            name: importlib.metadata.version(name)
            for name in (
                "alex-quant-lab",
                "numpy",
                "pandas",
                "lightgbm",
                "scikit-learn",
                "exchange-calendars",
                "pydantic",
                "httpx",
            )
        }
        manifest = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source": source,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": dependencies,
            "source_sha256": source_hashes,
            "files": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(directory.iterdir())
            },
        }
        write_json(directory / "manifest.json", manifest)
        output.mkdir(parents=True, exist_ok=True)
        # Publish complete files on the same filesystem; manifest is the final completion marker.
        for path in sorted(directory.iterdir(), key=lambda item: item.name == "manifest.json"):
            os.replace(path, output / path.name)
