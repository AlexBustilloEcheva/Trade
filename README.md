# Alex Quant Lab

MVP funcional de investigación cuantitativa para acciones y ETF estadounidenses.
Descarga OHLCV diario, calcula variables causales, entrena un StockRanker global
LightGBM y simula una cartera de hasta cinco acciones contra momentum de 120
sesiones, SPY, todo el universo equiponderado y efectivo. El segundo incremento
alinea el objetivo con las aperturas de rebalanceo y limita concentración por
activo y sector. No contiene integración de órdenes.

## Instalación y primera ejecución sin credenciales

Requiere [uv](https://docs.astral.sh/uv/getting-started/installation/).
Los comandos se ejecutan desde la raíz del repositorio. `uv` instala Python 3.12
si hace falta; la primera instalación necesita red. `uv.lock` fija también las
dependencias transitivas. NumPy se limita a 2.2 por compatibilidad con pandas 2.x.

```shell
uv sync --frozen
uv run --frozen alex-quant run --config configs/fixture.toml --provider fixture
uv run --frozen pytest
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

El modo `fixture` es **explícitamente sintético**: verifica el mismo flujo con
LightGBM real, sin claves ni red. Su informe y manifiesto lo identifican; sus
rentabilidades no son evidencia de una estrategia rentable. Las pruebas bloquean
conexiones de red y usan fixtures deterministas y respuestas HTTP simuladas.

Alternativa sin uv, usando Python 3.12 y un entorno virtual:

```shell
python -m venv .venv
.venv/Scripts/python -m pip install -e . pytest ruff
.venv/Scripts/python -m alex_quant.cli run --config configs/fixture.toml --provider fixture
```

Esta alternativa usa los rangos de `pyproject.toml`; use uv y el lock para reproducir
exactamente las versiones. En Linux/macOS, cambie `.venv/Scripts/python` por
`.venv/bin/python`.

## Datos reales de Alpaca

Configure `ALPACA_API_KEY` y `ALPACA_SECRET_KEY` en su entorno o en un archivo
`.env` local excluido de Git. El proceso carga ese archivo únicamente al usar Alpaca;
los modos fixture/snapshot y CI no lo leen. Para el backtest real ejecute:

```shell
uv run --frozen alex-quant run --config configs/default.toml
```

Las únicas variables secretas son `ALPACA_API_KEY` y `ALPACA_SECRET_KEY`.
También pueden exportarse en el entorno; tienen prioridad sobre `.env`.
No se incluyen credenciales, no se escriben en los artefactos ni se imprimen las
respuestas de error del servidor. `.env` está excluido de Git.

Antes de descargar todo el universo, diagnostique **solo SPY** durante pocos días:

```shell
uv run --frozen alex-quant diagnose-data --start 2025-01-06 --end 2025-01-10 --feed sip
```

El diagnóstico exige entre 2 y 5 sesiones históricas dentro de un máximo de 11 días
naturales. Valida el calendario y OHLCV y devuelve un JSON con feed, número de barras
y fechas; no escribe datos ni muestra claves. Reutiliza el proveedor de Market Data,
sin endpoints de trading ni cambio automático de feed. Con claves ausentes falla
claramente; sus respuestas correctas y sus errores también se verifican con mocks.

Sin claves, el comando termina con código 2 y explica cómo configurarlas; nunca
reemplaza datos reales por fixtures. HTTP 401/403 explica cómo revisar credenciales
y permisos; HTTP 429/5xx y fallos de conexión tienen hasta tres intentos.

El proveedor implementa `MarketDataProvider` mediante la API REST de datos de
Alpaca, con paginación completa, fechas locales de Nueva York y horario de verano.
Solicita ajustes `all` y mapeo de símbolos `asof=data.end`. El feed predeterminado
es SIP; debe estar autorizado para su cuenta y rango de fechas. IEX se puede elegir
en el TOML, pero es una sola bolsa y sus volúmenes y precios pueden diferir.
No se cambia de feed automáticamente. `data.end` debe ser anterior a hoy.

Referencias: [barras históricas de Alpaca](https://docs.alpaca.markets/us/reference/stockbarsingle-1),
[feeds y cobertura](https://docs.alpaca.markets/us/docs/market-data-faq).

## Configuración

`configs/default.toml` contiene los 20 valores solicitados y sus ETF sectoriales:
XLK, XLC, XLY, XLP, XLF, XLV, XLI y XLE. También descarga SPY y QQQ. QQQ se conserva
como referencia de datos; la comparación del primer MVP utiliza SPY.

| Sección | Parámetros principales |
|---|---|
| `data` | `start`, `end` inclusivos, `feed`, `benchmarks`; calendario XNYS y ajustes all |
| `universe` | Relación `ACCIÓN = "ETF_SECTORIAL"` |
| `research` | `label_mode`, `horizon_sessions`, `minimum_history_sessions`, `seed`, `rebalance_frequency` |
| `allocation` | `top_k`, `max_asset_weight` (20%), `max_sector_weight` (40%) |
| `walk_forward` | `train_sessions`, `min_train_sessions`, `retrain_every_signals`, `embargo_sessions` |
| `portfolio` | Capital, `commission_bps`, `slippage_bps`, `cash_annual_rate`, `annualization` |
| `model` | Árboles, learning rate, hojas, mínimo por hoja y regularización |

La frecuencia admite `W-FRI` (última sesión de la semana terminada en viernes),
`W-MON`, `daily` y `monthly`. Un festivo desplaza la señal a la última sesión del
periodo; un periodo truncado al final de los datos no genera una señal adelantada.
La ejecución ocurre en la siguiente sesión. Las ventanas se cuentan en sesiones,
no días naturales. El mínimo de historia no puede ser menor que 120.

`configs/fixture.toml` hereda de `default.toml` y reduce fechas y entrenamiento.
Puede crear otros experimentos con `extends = "default.toml"`. Las secciones se
fusionan salvo `universe`, que se reemplaza completa para evitar símbolos sobrantes.
Los parámetros desconocidos y las configuraciones inválidas producen un error.

El modo predeterminado en `default.toml` es `rebalance_open_to_open`: solo se
etiquetan las fechas de señal, con entrada en la apertura siguiente y salida en
la apertura posterior a la siguiente señal programada. Se guarda `entry_date` y
`label_end`; sin salida observada el objetivo es ausente. No se amplía el intervalo
de precios ni se cierra una etiqueta en el último día disponible.

El modo anterior `close_to_close` conserva `horizon_sessions=10`; este parámetro
**solo tiene efecto en ese modo**. `configs/legacy.toml` permite compararlo:

```shell
uv run --frozen alex-quant run --config configs/legacy.toml --output artifacts/legacy
```

`train_sessions` y `min_train_sessions` siguen contando fechas distintas de muestras
etiquetadas elegibles, no filas por acción ni días naturales: en modo alineado
semanal cuentan señales semanales (104/52 por defecto; 20/12 en el fixture).
En modo antiguo cuentan sesiones diarias (504/252 en `legacy.toml`). Se conserva el
calentamiento de 120 sesiones de precios. Estos tamaños dan aproximadamente dos
años/un año de muestras semanales y no se eligieron para mejorar rentabilidades.

La clave antigua `research.top_k` se migra a `allocation.top_k`; indicar ambas
produce error, incluso si coinciden. Una configuración antigua sin `label_mode`
mantiene el objetivo cierre a cierre, pero usa el nuevo asignador por defecto.
Para reproducir exactamente una ejecución del primer MVP hay que conservar su
versión de código. Las configuraciones resueltas de este incremento guardan el
modo y la asignación explícitos.

El asignador recorre el ranking (empates por ticker). Cada candidato recibe el
mínimo entre `1/top_k`, el límite por activo, la capacidad restante de su sector
y el capital restante. Un sector lleno se salta; una asignación parcial cuenta
como posición. No redistribuye huecos: el resto queda en efectivo. Por ejemplo,
si todos pertenecen al mismo sector, como máximo se asignan dos posiciones del
20% y el 60% queda en efectivo. Un score negativo sigue siendo elegible: es exceso
sectorial estimado, no rentabilidad absoluta ni probabilidad. Los límites son
hipótesis de investigación, no valores óptimos demostrados.

## Resultados y reproducción

Por defecto se escriben o reemplazan archivos en `artifacts/latest/`:

```text
metrics.json             métricas netas y concentración de las cinco carteras
equity.csv               patrimonio y contabilidad diaria; incluye la base inicial
predictions.csv          scores, ranking, selección, fechas y modelo de las 20 acciones
positions.csv            tenencias y pesos diarios al cierre, por cartera
report.md                informe, metodología, limitaciones y tabla comparativa
equity.svg               gráfico local sin servicios externos
trades.csv               nominal, apertura de referencia y costes por operación
input_bars.csv           snapshot OHLCV validado, con precisión completa
config.json              configuración resuelta
training_windows.json    fechas de entrenamiento, fin de etiquetas y bloques de predicción
models.json              modelos LightGBM serializados como texto
manifest.json            procedencia, hashes SHA-256, versiones, plataforma y hora UTC
labels.csv               objetivos, entry_date y label_end; incluye filas sin etiqueta
schedule.csv             calendario compartido de señales, entradas y salidas observadas
momentum_predictions.csv ranking y asignación del benchmark de 120 sesiones
yearly.csv               rentabilidad y drawdown por año (incluidos años parciales)
sector_exposure.csv      pesos diarios por sector y cartera
information_coefficient.csv Spearman por fecha, solo con etiquetas realizadas
metrics_stress.json      métricas con comisión y deslizamiento duplicados
equity_stress.csv        curva diaria del escenario de costes x2
trades_stress.csv        operaciones y costes del escenario x2
positions_stress.csv     posiciones del escenario x2
```

Para conservar varias ejecuciones, utilice `--output artifacts/nombre`.
Todos los resultados están excluidos de Git. El primer comando con fixtures también
usa `artifacts/latest`; compruebe siempre la procedencia del informe.

Repetir una ejecución guardada sin volver a descargar ni necesitar claves:

```shell
uv run --frozen alex-quant run --config artifacts/latest/config.json --provider snapshot --snapshot artifacts/latest --output artifacts/replay
```

Se verifica el hash del snapshot y se conserva su designación real/sintética.
La prueba de integración exige igualdad **byte a byte** de todos los resultados
de una ejecución y su repetición, salvo el manifiesto, que registra otra hora.
Para reproducibilidad fuera de esta máquina, conserve también el código y `uv.lock`.
LightGBM usa CPU, un hilo, semilla fija y modo determinista; versiones o plataformas
distintas aún pueden producir diferencias numéricas.

## Método y garantías comprobadas

- Variables: retornos 5/20/60/120; volatilidad 20/60; distancia al máximo de cierres
  20/60; drawdown desde el inicio del histórico; volumen relativo; fuerza relativa
  frente a SPY y ETF sectorial a 20/60 sesiones; beta frente a SPY a 60 sesiones.
- Etiqueta principal: exceso sectorial apertura a apertura entre ejecuciones
  programadas. Las últimas filas sin objetivo realizado pueden predecirse.
- Un regresor global, sin escaladores ajustados al futuro, se reentrena por bloques.
  Solo admite etiquetas completamente conocidas **antes** de la primera predicción
  del bloque, más el embargo opcional. No hay particiones aleatorias ni tuning.
- Cartera de hasta cinco posiciones, limitada al 20% por activo y al 40% por sector
  al ejecutar; los pesos después derivan con los precios. El informe mide los excesos
  diarios sin forzar operaciones fuera del calendario. Se usan fracciones de acción
  y se descuentan costes antes de invertir, sin saldo negativo ni apalancamiento.
- Momentum utiliza exclusivamente el retorno conocido de 120 sesiones y el mismo
  asignador, calendario, costes y periodo de evaluación que el modelo.
- SPY buy-and-hold y la cartera equiponderada pagan los mismos costes por nominal;
  todas las carteras tienen el mismo periodo y capital inicial. Efectivo al 0%
  por defecto; su tasa configurable también es la referencia de Sharpe/Sortino.
- Se rechazan duplicados, orden incorrecto, NaN, precios/volúmenes inválidos y barras
  ausentes, incluso si falta una sesión completa para todos los símbolos.

Las pruebas cubren causalidad de variables y predicciones, desplazamiento de etiquetas,
purga temporal, sector relativo, límites de cartera, costes, ejecución en la próxima
apertura, métricas calculadas a mano, calendario/festivos, errores de Alpaca y replay.
Los scripts `scripts/check.ps1` y `scripts/check.sh` ejecutan lint, formato y pytest.
GitHub Actions ejecuta también `uv sync --frozen` y el pipeline sintético, sin
claves ni llamadas a Alpaca. El informe incluye retorno/drawdown por año, efectivo,
concentración y deriva de límites, IC de Spearman por fecha madura y costes x2
con las mismas predicciones. No se reentrena para el escenario de estrés.

Consulte [docs/methodology.md](docs/methodology.md) para las fórmulas, convenciones
contables y el alcance preciso de las garantías temporales.

## Limitaciones y siguiente incremento

Es un laboratorio de investigación, sin ejecución real ni paper trading. Usa un
universo fijo elegido hoy y sectores fijos, con sesgo de supervivencia. Los snapshots
actuales de Alpaca pueden contener revisiones: no son un archivo histórico
point-in-time. Los ajustes corporativos son una aproximación a retorno total;
no hay contabilidad explícita de pagos de dividendos, escisiones o bajas.
El modo de comparación cierre a cierre conserva su desajuste con la ejecución;
el modo principal lo corrige usando aperturas entre rebalanceos. La etiqueta es
bruta y la rentabilidad de cartera es neta de costes. No hay validación económica
concluyente ni intervalos de confianza.

El siguiente incremento debería mejorar la validez histórica: universo y sectores
point-in-time, registro de acciones corporativas y calidad de datos, seguido de
validación anidada purgada y evaluación de estabilidad fuera de muestra. Los límites
20%/40% y el escenario de costes x2 son supuestos iniciales de investigación.
No añadir ejecución hasta resolver esa base de investigación.
