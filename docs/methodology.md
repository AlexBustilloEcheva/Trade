# Metodología del primer MVP

## Reloj de información

Cada fila representa una sesión de Nueva York. La barra completa del día `t` solo
se considera conocida después de su cierre. Las variables incluyen ese cierre
y volumen; la primera ejecución permitida es la apertura de `t+1`.
El calendario XNYS determina festivos y sesiones, también para validar huecos.

El objetivo es:

```text
y(stock,t) = close(stock,t+h)/close(stock,t) - close(sector,t+h)/close(sector,t)
label_end(t) = sesión t+h
```

`h=10` por defecto. Es un exceso de retorno simple, no una ratio ni un spread de
precios. La etiqueta usa información futura deliberadamente; nunca se emplea como
variable. Al ajustar un modelo para la señal `s`, solo entran filas cuya
`label_end < sesión(s - embargo_sessions)`. Con embargo cero también se excluyen
etiquetas que terminan en `s`, una convención conservadora.

Se toman las últimas `train_sessions` fechas elegibles y se exige
`min_train_sessions` fechas distintas, **no** filas multiplicadas por símbolos.
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

La comisión y el deslizamiento se registran por separado. El deslizamiento es un
coste proporcional al nominal de apertura, no un modelo de precio límite o de
impacto. Los pesos seleccionados son `1/top_k` sobre capital neto. Se admiten
fracciones de acción y no se permite endeudamiento. La prueba con precios conocidos
comprueba conservación del capital y costes, incluida una rotación total de cartera.

SPY se compra una sola vez en la misma primera apertura. La cartera equiponderada
incluye las 20 acciones y rebalancea en las fechas del StockRanker, con los mismos
costes. La cartera de efectivo no opera. No se liquida al final del backtest;
el valor terminal es marcado al último cierre y no incluye costes de liquidación.
Los splits y dividendos están representados por ratios de precios ajustados, no
por números de acciones legales ni flujos de pago explícitos.

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

## Reproducibilidad y alcance

`input_bars.csv` conserva 17 dígitos significativos; se lee con `float_precision=round_trip`.
La procedencia sintética se conserva al repetir un snapshot. El manifiesto contiene
hashes de los archivos, de cada módulo Python y versiones de dependencias. Los
archivos se preparan en un temporal y se reemplazan individualmente; el manifiesto
se publica al final. No ejecute dos procesos simultáneamente sobre la misma salida.

LightGBM usa un hilo CPU, semilla fija, `deterministic=True` y `force_col_wise=True`,
de acuerdo con sus [parámetros oficiales](https://lightgbm.readthedocs.io/en/stable/Parameters.html).
La reproducibilidad exacta se verifica dentro del entorno bloqueado; no se promete
identidad numérica entre distintas bibliotecas, compiladores o arquitecturas.

El universo no es point-in-time y presenta sesgo de supervivencia. Las etiquetas
se solapan entre sesiones y no coinciden exactamente con el periodo real de
tenencia. No hay optimización de parámetros, selección de modelos, intervalos de
confianza ni pruebas de significación. Tampoco hay restricciones sectoriales,
impacto dependiente de liquidez, impuestos ni eventos de exclusión bursátil.
