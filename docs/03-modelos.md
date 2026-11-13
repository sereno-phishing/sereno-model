# Fase 3: modelos

Estado: abierta el 2026-11-12.

## Objetivo

Entrenar el ensemble Stacking del ADD (Random Forest, XGBoost y LightGBM como modelos base; regresión logística como meta-modelo) sobre las matrices de la Fase 2, evaluarlo una sola vez en la prueba ciega y publicar un artefacto versionado con su reporte, listo para que `sereno-backend` lo cargue.

## Requisitos del ADD

| Requisito | Contenido | Origen |
|---|---|---|
| Modelos | RF, XGBoost y LightGBM como base; regresión logística como meta-modelo; bibliotecas compatibles con la interfaz de scikit-learn | Tabla 10, sección 8.1 |
| Salida | Veredicto de dos clases (phishing o legítima), puntaje entre 0 y 1 y versión del modelo | RF-05 |
| Meta | F1 >= 0.96 en la prueba ciega con la partición fija | Tabla 11, OE4-I1, RC-01 |
| Otras métricas | Precisión y recall se reportan sin umbral propio; exactitud como referencia | Tabla 11 |
| Latencia | Promedio <= 200 ms de extremo a extremo, HIT y MISS por separado | RNF-01, Tabla 13 |
| Artefacto | Un archivo por versión, con identificador, fecha y reporte de métricas; reversión en menos de 5 min | Tablas 5, 6 y 13 |
| Reproducibilidad | Misma partición y semilla | Tabla 12 |
| Entrenamiento | Fuera de línea, en equipo propio, sin GPU | Tabla 3 |

El ADD no fija cómo se producen las entradas del meta-modelo, la validación cruzada, el ajuste de hiperparámetros, el umbral, la calibración ni el formato del archivo. Esas decisiones se toman en esta fase, con sustento en el corpus.

## Decisiones heredadas

- Partición 80/20 aleatoria y estratificada, con semilla 20261111, como en los papers; la agrupada por dominio registrado queda como análisis de robustez (Fase 1, registro del 2026-11-12).
- Sin remuestreo adicional: las clases ya están balanceadas (Fase 1).
- Entrenamiento oficial en CPU (`decisiones.md`).
- Vector versión 2, 24 características; las columnas `audit_*` no entran al modelo (Fase 2). El modelo oficial les suma un TF-IDF de caracteres de la URL (ver «Método aprobado»).
- Quedan para esta fase: evaluación entre fuentes como P20, métricas aparte para plataformas compartidas y acortadores, e importancia y estabilidad de las características.

## Entradas

- `data/processed/20261112T233800Z/features_train.parquet`: 64,424 filas (32,212 phishing y 32,212 legítimas), hash `cd449ecc…a169`. Partición aleatoria `20261112T232911Z`; matrices generadas desde el commit `1df9c0a` con el repositorio limpio.
- `data/processed/20261112T233800Z/features_test.parquet`: 16,106 filas (8,053 y 8,053). Solo se usa en la evaluación final.
- Columna `group` (dominio registrado), para los pliegues agrupados del análisis de robustez y para el análisis de errores.
- Las corridas 1.0.0 y de diagnóstico usaron las matrices `20261112T025607Z` de la partición agrupada anterior.

## Método aprobado

Aprobado por Andrés el 2026-11-12 y actualizado el mismo día con la variante C del TF-IDF.

| Decisión | Propuesta | Sustento |
|---|---|---|
| Entradas de los modelos base | Las 24 características v2 más un TF-IDF de caracteres del texto medido de la URL: n-gramas de 1 a 3, `max_features=5000`, norma l2, IDF suavizado, sin `sublinear_tf`. El vectorizador se ajusta dentro de cada pliegue (Pipeline) y la matriz es dispersa | `max_features`: P02. Parámetros del transformador: P32. Rango de n-gramas: decisión propia medida en la ablación `artifacts/diagnostics/20261112T192824Z` |
| Entradas del meta-modelo | Probabilidades de cada modelo base predichas por pliegue (fuera de pliegue) sobre el entrenamiento | P15 (super learner por pliegues); P27 (regresión logística como meta-modelo) |
| Validación cruzada | 5 pliegues estratificados y aleatorios (`StratifiedKFold`, semilla 20261111), según la estrategia de la partición; los agrupados por dominio quedan como análisis de robustez (`--cv-strategy grouped`) | 5 pliegues: P09, P07. Pliegues aleatorios: P02 y los demás papers del corpus |
| Ajuste de hiperparámetros | Búsqueda aleatoria con los rangos de P09, evaluada con F1 en la validación cruzada | P09 (RandomizedSearch y rangos); P07, P17, P20 y P24 usan búsqueda en rejilla |
| Umbral | Elegido en la validación cruzada, el de mayor F1 promedio; nunca con la prueba | P17 |
| Calibración | No se aplica | El corpus no la trata |
| Formato del artefacto | `joblib`, un archivo por versión, con su `report.json`. El modelo recibe la URL cruda y calcula el texto medido y el vector v2 | Decisión menor de implementación |

## Criterio de salida

- Artefacto y `report.json` en `artifacts/<version>/`, con commit, semilla, hashes de datos y partición, versiones de librerías, parámetros, métricas y duración.
- F1, precisión, recall y exactitud en la prueba ciega, medidos una sola vez.
- Evaluación entre fuentes y métricas para plataformas compartidas y acortadores.
- Explicación de la fase por Andrés, con sus palabras.

## Registro

### 2026-11-12: apertura

Se reunieron los requisitos del ADD, las decisiones heredadas y lo que hace el corpus en cada decisión abierta. La propuesta de método espera la aprobación de Andrés antes de escribir código.

### 2026-11-12: código de entrenamiento (T1)

Se escribió el paquete `src/sereno_model/models/` con el método aprobado, sin corrida oficial:

- `stacking.py`: selección de las 24 características (falla si la matriz trae columnas ajenas o le falta alguna), pliegues agrupados por dominio, búsqueda aleatoria con F1, probabilidades fuera de pliegue, meta-modelo, umbral y la clase `StackingModel` que se guarda como artefacto.
- `train.py`: lee solo `features_train.parquet`, comprueba su hash contra `features_report.json` y escribe `model.joblib` y `report.json` en `artifacts/<version>/`. No sobrescribe una versión existente y se niega a correr con cambios sin commit, salvo con `--allow-dirty`, que marca el reporte como exploración no citable.
- Uso: `python -m sereno_model.models.train --version <version>`.

El artefacto expone `score` (probabilidad de phishing del meta-modelo, entre 0 y 1), `predict` (1 si el puntaje alcanza el umbral), el umbral, los nombres y la versión de las características y la versión del modelo.

Decisiones propias de esta tarea:

- `n_iter` = 20 candidatos por modelo base. El corpus da los rangos de búsqueda, no el número de candidatos.
- Tasas de aprendizaje de 0.05 a 0.20 en pasos de 0.01, porque `scipy` no es una dependencia declarada.
- Umbrales candidatos de 0.01 a 0.99 en pasos de 0.01; ante empate gana el más cercano a 0.5.
- Cada estimador corre con un hilo y la paralelización reparte ajustes completos entre procesos, así el resultado no depende del número de procesos.
- Las métricas de validación cruzada del Stacking se calculan con el umbral elegido en esos mismos pliegues, por lo que son algo optimistas; la cifra que se reporta en la tesis es la de la prueba ciega.

La prueba ciega no se lee en esta tarea. Las pruebas automáticas usan matrices sintéticas pequeñas con el esquema real.

### 2026-11-12: corrida oficial de entrenamiento 1.0.0

Comando: `python -m sereno_model.models.train --version 1.0.0 --n-jobs 16`, desde el commit `6da51eb` con el repositorio limpio. Semilla 20261111; matriz de entrenamiento con hash `308d6b88…41eb`. Duración: 160 s. Reporte en `artifacts/1.0.0/report.json`.

| Modelo | F1 en validación cruzada | Hiperparámetros elegidos |
|---|---|---|
| Random Forest | 0.8459 | 62 árboles, profundidad 10 |
| XGBoost | 0.8732 | 54 árboles, profundidad 9, tasa 0.19 |
| LightGBM | 0.8716 | 89 árboles, profundidad 9, tasa 0.18 |
| Stacking | 0.8763 (precisión 0.8704, recall 0.8826, exactitud 0.8755) | Umbral 0.37 |

El F1 en validación cruzada queda por debajo de la meta del ADD (0.96). La prueba ciega no se usó.

### 2026-11-12: diagnóstico de la brecha con la meta (exploración)

Corrida de exploración, no citable como resultado final. Comando: `python -m sereno_model.models.diagnose --n-jobs 16`, desde el commit `6da51eb` con cambios sin commit (`reportable: false`). Solo la matriz de entrenamiento (hash `308d6b88…41eb`), semilla 20261111, 5 pliegues, 20 candidatos por búsqueda. Duración: 757 s. Reporte en `artifacts/diagnostics/20261112T153735Z/report.json`. Código: `src/sereno_model/models/diagnose.py`; la parte de validación cruzada de `stacking.py` se separó en `cross_validate` para reutilizarla sin cambiar el resultado.

**1. Efecto de la agrupación por dominio.** Mismos rangos de P09. La variante agrupada reproduce la corrida 1.0.0 (mismos hiperparámetros, F1 0.8763, umbral 0.37).

| F1 en validación cruzada | Random Forest | XGBoost | LightGBM | Stacking (umbral) |
|---|---|---|---|---|
| Pliegues agrupados por dominio | 0.8459 | 0.8732 | 0.8716 | 0.8763 (0.37) |
| Pliegues sin agrupar | 0.8620 | 0.8900 | 0.8868 | 0.8923 (0.43) |

Stacking sin agrupar: precisión 0.8979, recall 0.8868, exactitud 0.8930.

**2. Efecto de los rangos.** Pliegues agrupados con rangos propios más amplios, sin sustento en el corpus: RF 100 a 500 árboles, profundidad {10, 20, 30, sin límite}, 1 a 5 muestras por hoja; XGBoost 100 a 600 árboles, profundidad 4 a 12, tasa 0.03 a 0.30; LightGBM 100 a 600 árboles, profundidad {sin límite, 8 a 16}, 31 a 255 hojas, tasa 0.03 a 0.30.

| Modelo | F1 con P09 | F1 con rangos amplios | Hiperparámetros elegidos (rangos amplios) |
|---|---|---|---|
| Random Forest | 0.8459 | 0.8687 | 111 árboles, profundidad 20, 2 muestras por hoja |
| XGBoost | 0.8732 | 0.8738 | 234 árboles, profundidad 7, tasa 0.08 |
| LightGBM | 0.8716 | 0.8727 | 302 árboles, profundidad 9, 32 hojas, tasa 0.13 |
| Stacking | 0.8763 | 0.8773 (umbral 0.41) | |

**3. Análisis de errores** del Stacking agrupado con rangos de P09, sobre sus puntajes fuera de pliegue con el umbral 0.37: 28,429 VP, 4,239 FP, 27,972 VN y 3,783 FN (tasa de falsos positivos 0.1316).

| Subgrupo | Filas | Phishing | F1 | Precisión | Recall | Tasa de FP |
|---|---|---|---|---|---|---|
| Fuente PhishTank | 21,322 | 21,322 | 0.9466 | 1.0000 | 0.8986 | n. a. |
| Fuente url_phish_2026 | 37,063 | 8,787 | 0.7924 | 0.7120 | 0.8934 | 0.1123 |
| Fuente hannousse_2021 | 7,477 | 4,446 | 0.7957 | 0.7838 | 0.8079 | 0.3270 |
| Fuente Majestic Million (= filas sintéticas) | 906 | 0 | n. a. | n. a. | n. a. | 0.0806 |
| Dominios recortados por el tope (plataformas y acortadores) | 6,053 | 6,000 | 0.9879 | 0.9948 | 0.9812 | 0.5849 (53 legítimas) |
| Resto de dominios | 58,370 | 26,212 | 0.8513 | 0.8427 | 0.8600 | 0.1309 |

Una fila con varias fuentes cuenta en cada una; hay 2,345 filas así. Los dominios recortados son los 16 que registra `assemble_report.json`; no existe en el código otra lista de acortadores o plataformas compartidas.

Dominios registrados más frecuentes entre los errores (solo conteos, sin URL):

- Falsos positivos: `wordpress.com` 99, `academic.ru` 50, `inrae.fr` 38, `tumblr.com` 38, `techtarget.com` 37, `youtube.com` 32, `facebook.com` 27, `unistra.fr` 21, `univ-grenoble-alpes.fr` 20, `wikipedia.org` 19.
- Falsos negativos: `google.com` 70, `linkin.bio` 66, `ln.run` 66, `flowcode.com` 58, `urlz.fr` 37, `tinyurl.com` 30, `web.app` 28, `mytemp.website` 27, `ghost.io` 24, `linqapp.com` 20.

Correlación entre modelos base (probabilidades fuera de pliegue; errores con umbral 0.5):

| Par | Correlación de probabilidades | Correlación de errores |
|---|---|---|
| RF y XGBoost | 0.946 | 0.706 |
| RF y LightGBM | 0.947 | 0.717 |
| XGBoost y LightGBM | 0.986 | 0.859 |

De 64,423 filas, 53,119 no las falla ningún modelo base, 3,317 las falla uno, 2,038 dos y 5,949 los tres.

Lectura: la agrupación por dominio resta 0.016 de F1 al Stacking y los rangos amplios suman 0.001; ninguna de las dos acerca la cifra a 0.96. Los modelos base cometen errores muy parecidos, en especial XGBoost y LightGBM, y 5,949 filas las fallan los tres. Las fuentes con ambas clases (url_phish_2026 y hannousse_2021) quedan cerca de 0.79 de F1, mientras PhishTank, que solo aporta phishing, llega a 0.95. Los falsos negativos se concentran en acortadores y plataformas de alojamiento; los falsos positivos, en blogs, sitios académicos y redes sociales.

### 2026-11-12: medición de las características de lista (exploración)

Ver `02-caracteristicas.md`, registro del 2026-11-12: con los tres grupos el F1 del Stacking en validación cruzada pasa de 0.8763 a 0.8821; la adopción queda pendiente de Andrés.

### 2026-11-12: TF-IDF de caracteres (exploración)

Informe `artifacts/diagnostics/20261112T192824Z/` (no citable: árbol con cambios sin commit). TF-IDF de caracteres según P02 (`analyzer='char'`, `max_features=5000`), ajustado dentro de cada pliegue. Búsqueda con 12 candidatos por modelo en las variantes con TF-IDF y 20 en A. Duración total: 7,300 s.

| Variante | Pliegues | RF | XGBoost | LightGBM | Stacking | Umbral | Diferencia con A |
|---|---|---|---|---|---|---|---|
| A: 24 características v2 | aleatorios | 0.8620 | 0.8900 | 0.8868 | 0.8923 | 0.43 | — |
| B: solo TF-IDF (n 1-3) | aleatorios | 0.8584 | 0.9398 | 0.9465 | 0.9476 | 0.43 | +0.0552 |
| C: v2 + TF-IDF (n 1-3) | aleatorios | 0.8702 | 0.9464 | 0.9506 | 0.9519 | 0.50 | +0.0595 |
| D: v2 + TF-IDF (n 3-5) | aleatorios | 0.8715 | 0.9385 | 0.9459 | 0.9465 | 0.41 | +0.0541 |
| E: C con pliegues agrupados | agrupados por dominio | 0.8463 | 0.9313 | 0.9357 | 0.9381 | 0.42 | +0.0458 |

Variante C: latencia de inferencia p50 6.0 ms y p95 6.9 ms por URL (RNF-01: ≤ 200 ms); modelo de 2.3 MB.

Lectura: C es la mejor (0.9519) y queda a 0.008 de la meta de 0.96. El TF-IDF aporta casi toda la mejora; las 24 características v2 suman poco encima. Con pliegues agrupados la mejora se mantiene (+0.046), así que no depende de que un mismo dominio aparezca en entrenamiento y validación. El Stacking supera a cada modelo base en todas las variantes. RF no aprovecha el TF-IDF.

### 2026-11-12: método oficial con TF-IDF y partición aleatoria

Decisión de Andrés sobre la ablación `artifacts/diagnostics/20261112T192824Z`:

- La partición pasa a ser aleatoria y estratificada, como en los papers (P02 y otros usan particiones aleatorias). La agrupada por dominio se conserva como análisis de robustez. Ver `01-datos.md`, registro del 2026-11-12.
- El modelo oficial es la variante C: 24 características v2 más TF-IDF de caracteres con n-gramas de 1 a 3. En la ablación obtuvo F1 0.9519 con pliegues aleatorios y 0.9381 con pliegues agrupados por dominio (variante E).

Cambios en el código:

- `stacking.py` reúne los bloques del TF-IDF que antes estaban en `tfidf_ablation.py` (texto medido, vectorizador, unión con el vector, pipeline por modelo base, medición de latencia y tamaño) y la ablación los importa de ahí. `fit_stacking` entrena por defecto la variante C con pliegues aleatorios; `ngram_range=None` y `fold_strategy="grouped"` reproducen el método de 1.0.0.
- `StackingModel` acepta una URL, una lista de URL o una tabla con la columna `url`, y calcula el texto medido y el vector v2. `score`, `predict_proba` y `predict` se mantienen; `score_url` puntúa una URL. Si la tabla trae las 24 columnas del vector, las usa.
- `train.py` toma la estrategia de pliegues de la partición registrada en `features_report.json` (las particiones anteriores, sin ese dato, eran agrupadas), salvo que `--cv-strategy` indique otra. El `report.json` guarda el conjunto de características con los parámetros del TF-IDF, la estrategia y el método de los pliegues, la latencia p50 y p95 por URL, el tamaño de `model.joblib`, las versiones, el commit y la semilla. Búsqueda por defecto: 20 candidatos por modelo base.

La corrida oficial no se hizo en esta tarea: corre en el equipo de Andrés y tarda horas.

### 2026-11-13: corrida oficial de entrenamiento 2.0.0

Comando: `python -m sereno_model.models.train --version 2.0.0 --n-jobs 4`, desde el commit `7c3255b` con el repositorio limpio. Semilla 20261111; matriz de entrenamiento `processed/20261112T233800Z` (64 424 filas, 32 212 por clase) con hash `cd449ecc…a169`. Pliegues aleatorios estratificados, tomados de la partición. Duración: 2948 s (búsqueda 2621 s). Reporte en `artifacts/2.0.0/report.json`.

| Modelo | F1 en validación cruzada | Hiperparámetros elegidos |
|---|---|---|
| Random Forest | 0.8781 | 62 árboles, profundidad 10 |
| XGBoost | 0.9440 | 54 árboles, profundidad 9, tasa 0.19 |
| LightGBM | 0.9484 | 89 árboles, profundidad 9, tasa 0.18 |
| Stacking | 0.9497 (precisión 0.9533, recall 0.9461, exactitud 0.9499) | Umbral 0.45 |

El F1 del Stacking por pliegue va de 0.9467 a 0.9526. El umbral se eligió sobre los mismos pliegues, así que estas cifras son algo optimistas. Coeficientes del metamodelo: Random Forest -1.27, XGBoost 3.10, LightGBM 6.19.

Latencia del modelo por URL, medida en 1000 URL de entrenamiento con un hilo: p50 6.884 ms, p95 7.659 ms (RNF-01 pide una media de 200 ms de extremo a extremo; esto mide solo el modelo). Tamaño de `model.joblib`: 2.86 MB.

El F1 en validación cruzada queda por debajo de la meta del ADD (0.96) y del valor de la ablación (0.9519). La prueba ciega no se usó.
