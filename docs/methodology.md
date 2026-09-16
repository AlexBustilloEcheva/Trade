# Metodología — investigación y validación v2

## Reloj de información

Cada fila representa una sesión de Nueva York. La barra completa del día `t` solo
se considera conocida después de su cierre. Las variables incluyen ese cierre
y volumen; la primera ejecución permitida es la apertura de `t+1`.
El calendario XNYS determina festivos y sesiones, también para validar huecos.

El modo principal, `rebalance_open_to_open`, usa únicamente fechas de señal `s`:

```text
entry_date(s) = apertura de la siguiente sesión después de s
label_end(s) = apertura de la siguiente sesión después de la próxima señal programada
y(stock,s) = open(stock,label_end)/open(stock,entry_date)
           - open(sector,label_end)/open(sector,entry_date)
```

`data/schedule.py` construye un calendario compartido por etiquetas, señales,
momentum y simulación. Detecta el final de cada periodo comparando sesiones
consecutivas del índice validado. No descarga ni utiliza barras fuera de `data.end`.
Un Viernes Santo desplaza la señal semanal al jueves y la entrada al lunes; una
semana truncada el miércoles no se considera terminada. Una señal sin entrada
observada no se ejecuta; una señal con entrada pero sin salida observada se puede
predecir y conserva objetivo/`label_end` ausentes. No se usa el último cierre como
salida artificial. Las sesiones que no son señales no tienen objetivo negociable.

Un salto de precio entre el cierre de la señal y su apertura de entrada queda
fuera del objetivo y del beneficio de la nueva posición. La prueba de regresión
duplica el precio antes de entrar y verifica que ese 100% no se capture.

El modo anterior `close_to_close`, disponible en `configs/legacy.toml`, conserva:

```text
y(stock,t) = close(stock,t+h)/close(stock,t) - close(sector,t+h)/close(sector,t)
label_end(t) = sesión t+h
```

`h=research.horizon_sessions`, 10 por defecto; este parámetro no controla el modo
alineado. En el modo antiguo `entry_date` representa la fecha del cierre inicial,
no una apertura. Ambos objetivos son excesos de retorno simple, no probabilidades
ni estimaciones de rentabilidad absoluta. La etiqueta usa información futura
deliberadamente; nunca se emplea como
variable. Al ajustar un modelo para la señal `s`, solo entran filas cuya
`label_end < sesión(s - embargo_sessions)`. Con embargo cero también se excluyen
etiquetas que terminan en `s`, una convención conservadora. En el modo apertura,
una salida en la apertura de `s` ya sería observable al cierre, pero también se
excluye para conservar la desigualdad estricta y el contrato de purga del proyecto.
El embargo se cuenta en sesiones bursátiles, incluso con observaciones semanales.

Se toman las últimas `train_sessions` fechas elegibles y se exige
`min_train_sessions` fechas distintas con objetivo conocido, **no** filas
multiplicadas por símbolos ni todas las sesiones transcurridas.
En modo alineado semanal son señales semanales; en modo cierre a cierre son fechas
diarias. Los valores del archivo principal pasan a 104/52 (unas dos temporadas
anuales de muestras, con mínimo de una); el fixture usa 20/12. `legacy.toml`
mantiene 504/252. El cambio de tamaños refleja la frecuencia de las muestras y
no una búsqueda de mejores resultados.
Al comienzo la ventana crece desde ese mínimo; luego rueda con tamaño limitado.
El modelo se mantiene congelado durante `retrain_every_signals` señales y después
se vuelve a ajustar desde cero. Una predicción pasada puede entrar en entrenamiento
de un bloque posterior una vez que su etiqueta ha madurado; esto es walk-forward,
no solapamiento entre entrenamiento y predicción de un mismo modelo.

Se guardan todas las fechas de entrenamiento, el máximo fin de etiqueta, el corte
exclusivo, el inicio/final del bloque de predicción y el número de muestras. El
ranking rompe empates por ticker para no depender del orden del proveedor.

## Variables

Con `P_t` cierre ajustado, `V_t` volumen ajustado, `r_t=P_t/P_(t-1)-1` y `A=252`:

| Variable | Definición |
|---|---|
| `return_k` | `P_t/P_(t-k)-1`, k=5,20,60,120 |
| `volatility_k` | Desviación muestral de las últimas k rentabilidades, multiplicada por sqrt(A), k=20,60 |
| `distance_high_k` | `P_t / max(P_(t-k+1)...P_t) - 1`, k=20,60; máximo de cierres |
| `drawdown` | `P_t / max(P_inicio...P_t) - 1` |
| `relative_volume_20` | `V_t / mean(V_(t-20)...V_(t-1))` |
| `relative_spy_k` | Retorno k del valor menos retorno k de SPY, k=20,60 |
| `relative_sector_k` | Retorno k del valor menos retorno k de su ETF, k=20,60 |
| `beta_spy_60` | Covarianza muestral acción/SPY dividida por varianza muestral de SPY, últimas 60 rentabilidades |

Ventanas completas, sin `center`, sin relleno de datos y sin normalización global.
Se requieren 120 retornos anteriores (121 cierres); el drawdown comienza en la fecha
inicial descargada, no en el máximo histórico de toda la vida del valor. Una beta
con varianza cero es indefinida y no se imputa. Los NaN de calentamiento se excluyen
del entrenamiento, pero no se eliminan las últimas filas solo por carecer de objetivo.
Las señales requieren variables válidas para todo el universo configurado.
Después de iniciar la evaluación, la falta de variables para una señal programada
produce un error: no se alarga silenciosamente la tenencia omitiendo el rebalanceo.

Las pruebas amplían y alteran el futuro sin cambiar las variables ni las predicciones
del pasado. También comprueban invariancia ante un reescalado uniforme de precios
y volumen: las variables no contienen niveles absolutos. Esto cubre causalidad del
código, **no** disponibilidad histórica de las revisiones del proveedor.

## Datos y acciones corporativas

El cliente solo accede a `https://data.alpaca.markets/v2/stocks/bars`.
Alpaca ordena las páginas por símbolo; el cliente consume `next_page_token` hasta
terminar y normaliza a `(date, symbol)`. La validación posterior exige la cuadrícula
completa de sesiones por símbolo y OHLCV coherente. No se toleran huecos silenciosos,
ni se rellena el precio de un activo suspendido o inexistente.

`adjustment=all` incluye ajustes soportados por Alpaca para splits, dividendos y
escisiones. `asof` identifica el mapeo de tickers a la fecha final; **no** significa
que las barras hayan sido archivadas en esa fecha ni que se excluyan revisiones
posteriores. La historia de nombres, escisiones (especialmente GE) y clasificaciones
del universo exige revisión antes de extraer conclusiones económicas.

Un ajuste multiplicativo posterior uniforme sobre todo el prefijo histórico
se cancela en las variables de ratios. Las correcciones del proveedor dentro de
las ventanas no tienen esa garantía. Una extensión con datos versionados y eventos
corporativos fechados es necesaria para un estudio point-in-time completo.

Fuentes: [API de barras y parámetros](https://docs.alpaca.markets/us/reference/stockbarsingle-1),
[paginación multiactivo](https://docs.alpaca.markets/us/v1.4.2/reference/stockbars),
[calendarios de mercado](https://github.com/gerrymanoim/exchange_calendars).

## Simulación de cartera

La curva comienza con el capital inicial al cierre de la primera señal aceptada.
En cada sesión posterior:

1. Las tenencias anteriores reciben `open_t/close_(t-1)`.
2. El efectivo anterior devenga la tasa efectiva por sesión.
3. En un rebalanceo se vende/compra en la apertura de referencia, descontando
   comisión y deslizamiento como costes en USD sobre todo el nominal negociado.
4. Las tenencias resultantes reciben `close_t/open_t`.

No se recibe el retorno nocturno de una acción antes de comprarla. Los pesos no se
resetean a diario: derivan entre las fechas de rebalanceo.

Para asegurar que las compras puedan financiarse, se resuelve por bisección:

```text
capital_post = capital_pre - tasa_costes * sum(abs(peso_objetivo * capital_post - tenencia_pre))
```

## Ranking y asignación independiente

`rank_scores` solo ordena scores de mayor a menor, desempatando por ticker.
`allocate` usa la sección independiente `[allocation]`, con `top_k=5`, máximo
por activo del 20% y máximo por sector del 40%.

Recorre el ranking y asigna a cada candidato:

```text
peso = min(1/top_k, max_asset_weight, max_sector_weight - peso_sector, 1 - invertido)
```

Si un sector está lleno, pasa al siguiente candidato sin consumir una posición.
Un peso parcial positivo sí consume una posición. Se detiene al alcanzar `top_k` y no
redistribuye los huecos. El capital restante queda en efectivo. Con `top_k=3` y
límite del 20%, puede invertir como máximo el 60% aunque haya muchos sectores.
Si todo el universo comparte sector, solo invierte el 40% con los valores iniciales.
Estos parámetros no son límites óptimos demostrados. No se filtra por signo del
score: un exceso sectorial positivo puede coexistir con una pérdida absoluta.

La clave antigua `research.top_k` se migra explícitamente a `allocation.top_k`,
sin cambiar su significado de número máximo de posiciones. Si existen ambas se
rechaza la configuración. Sin `label_mode`, un archivo antiguo conserva el objetivo
cierre a cierre. La asignación nueva sí se aplica salvo que se cambien sus límites;
reproducir el MVP antiguo completo requiere también su código original.

La comisión y el deslizamiento se registran por separado. El deslizamiento es un
coste proporcional al nominal de apertura, no un modelo de precio límite o de
impacto. Los pesos objetivo están definidos sobre capital neto después de costes. Se admiten
fracciones de acción y no se permite endeudamiento. La prueba con precios conocidos
comprueba conservación del capital y costes, incluida una rotación total de cartera.

SPY se compra una sola vez en la misma primera apertura. La cartera equiponderada
incluye las 20 acciones y rebalancea en las fechas del StockRanker, con los mismos
costes. Estas dos referencias se mantienen sin límites del asignador para no
desvirtuar su definición. Momentum sí usa idénticos límites y algoritmo que el
modelo: score igual a `close_t/close_(t-120)-1`, conocido al cierre, sin filtro de
signo ni información futura. Comparte universo, fechas, capital inicial y costes.
La cartera de efectivo no opera. No se liquida al final del backtest;
el valor terminal es marcado al último cierre y no incluye costes de liquidación.
Los splits y dividendos están representados por ratios de precios ajustados, no
por números de acciones legales ni flujos de pago explícitos.

Los límites se comprueban al aplicar los pesos objetivo en la apertura. La deriva
entre rebalanceos, incluso durante la sesión de ejecución, puede superarlos.
`equity.csv` registra exposición, peso del efectivo, máximos por activo/sector y
excesos sobre límites al cierre. `metrics.json` resume máximos, promedios y número
de sesiones con excesos; `sector_exposure.csv` permite inspeccionar cada sector.
Las tolerancias de comprobación son 1e-12 para excluir ruido de punto flotante.
SPY/equiponderación están identificados como referencias a las que no se aplican
los límites. No se disparan operaciones diarias para corregir deriva.

## Métricas

Se excluye la fila inicial de las observaciones de retorno, pero se incluye como
primer máximo para calcular drawdown. `N` son sesiones de rentabilidad. La tasa
libre de riesgo es la misma que la del benchmark de efectivo.

| Métrica | Convención |
|---|---|
| Acumulada | `equity_final/equity_inicial - 1` |
| Anualizada | `(equity_final/equity_inicial)^(A/N) - 1` |
| Volatilidad | `std(r, ddof=1) * sqrt(A)` |
| Sharpe | `mean(r-rf_diario)/std(r, ddof=1) * sqrt(A)` |
| Sortino | `mean(r-rf_diario)/sqrt(mean(min(r-rf_diario,0)^2)) * sqrt(A)` |
| Máximo drawdown | Mínimo de `equity/cummax(equity)-1`; número negativo |
| Calmar | Retorno anualizado dividido por valor absoluto del máximo drawdown |
| Rotación | Nominal comprado **más** vendido dividido por capital anterior a operar; sin factor 1/2 |
| Rotación anualizada | Suma de rotación multiplicada por `A/N` |
| Costes | Suma en USD, y fracción del capital inicial; incluye comisiones y deslizamiento |
| Rebalanceos | Número de fechas en las que se aplica un objetivo, incluida la entrada |
| Exposición media | Media diaria del valor de las acciones dividido por equity al cierre |

La desviación a la baja de Sortino usa **todas** las observaciones, con cero en las
positivas; no es la desviación estándar de solo las pérdidas. Cocientes con
denominador cero se guardan como `null`. `rf_diario=(1+rf_anual)^(1/A)-1`.
Las anualizaciones de periodos breves pueden ser poco representativas.

## Evaluación adicional

`yearly.csv` usa rentabilidades netas de cada año, sin anualizar periodos parciales.
El drawdown se calcula dentro del año comenzando en el valor anterior a la primera
sesión, para incluir también una caída del primer día. No es el drawdown global
arrastrado desde un pico de un año anterior.

`information_coefficient.csv` calcula, para modelo y momentum, Spearman por fecha
de señal: correlación Pearson entre los rangos de scores y objetivos realizados,
asignando el rango medio a los empates. Solo incluye la sección completa cuando
todas las etiquetas han madurado (`label_end <= última sesión observada`). Esta
desigualdad de evaluación usa la información disponible al cierre final y no cambia
la purga estricta de entrenamiento. Una sección constante o con menos de dos
activos tiene IC ausente, no cero. No se seleccionan solo las acciones compradas.
Las fechas sin etiqueta madura se excluyen, sin ampliar el rango de descarga.
El informe muestra cada fecha; estos IC no son pruebas de significación ni una
validación económica concluyente.

El escenario de estrés vuelve a simular las cinco carteras con las **mismas**
predicciones, pesos objetivo, calendario y datos. Duplica comisiones y deslizamiento
configurados sin reentrenar ni volver a seleccionar candidatos. Los nominales
cambian al reducirse el capital, por lo que el coste acumulado no tiene por qué
ser exactamente dos veces el coste base. Se exportan métricas, curva, operaciones
y posiciones del escenario. El caso base sigue intacto para reproducibilidad.

## Diagnóstico y CI

`alex-quant diagnose-data --start 2025-01-06 --end 2025-01-10 --feed sip` consulta
exclusivamente barras de SPY, con el mismo proveedor, variables de entorno, ajustes
y controles de datos. Limita el intervalo a entre 2 y 5 sesiones dentro de hasta
11 días naturales. Carga credenciales dentro del proceso Alpaca; no imprime `.env`,
claves, cabeceras ni cuerpos de errores. El JSON de éxito solo incluye estado,
símbolo, feed, fechas, número de barras y resultado de validación.

`.github/workflows/ci.yml` instala las versiones de `uv.lock` con `uv sync --frozen`,
ejecuta pytest, lint, formato y el pipeline sintético. No recibe secretos ni llama
a Alpaca; las pruebas bloquean conexiones y simulan respuestas del proveedor.
El flujo sigue la [integración oficial de uv con GitHub Actions](https://github.com/astral-sh/uv/blob/main/docs/guides/integration/github.md).

## Reproducibilidad y alcance

`input_bars.csv` conserva 17 dígitos significativos; se lee con `float_precision=round_trip`.
La procedencia sintética se conserva al repetir un snapshot. El manifiesto contiene
hashes de los archivos, de cada módulo Python y versiones de dependencias. Los
archivos se preparan en un temporal y se reemplazan individualmente; el manifiesto
se publica al final. La publicación hereda los permisos del directorio de salida,
incluido en Windows, en lugar de los permisos privados del directorio temporal.
No ejecute dos procesos simultáneamente sobre la misma salida.

LightGBM usa un hilo CPU, semilla fija, `deterministic=True` y `force_col_wise=True`,
de acuerdo con sus [parámetros oficiales](https://lightgbm.readthedocs.io/en/stable/Parameters.html).
La reproducibilidad exacta se verifica dentro del entorno bloqueado; no se promete
identidad numérica entre distintas bibliotecas, compiladores o arquitecturas.

El universo no es point-in-time y presenta sesgo de supervivencia. El modo antiguo
conserva etiquetas solapadas y desajuste entre objetivo y ejecución. En el modo
alineado, conservar un activo en dos rebalanceos enlaza dos intervalos de objetivo;
la etiqueta es bruta, mientras que la rentabilidad de cartera descuenta costes.
No hay optimización de parámetros, selección de modelos, intervalos de confianza
ni pruebas de significación. Tampoco hay impacto dependiente de liquidez, impuestos
ni tratamiento explícito de eventos de exclusión bursátil.
