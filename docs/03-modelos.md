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

- Partición 80/20 agrupada por dominio registrado y estratificada, con semilla 20261111 (Fase 1).
- Sin remuestreo adicional: las clases ya están balanceadas (Fase 1).
- Entrenamiento oficial en CPU (`decisiones.md`).
- Vector versión 2, 24 características; las columnas `audit_*` no entran al modelo (Fase 2).
- Quedan para esta fase: evaluación entre fuentes como P20, métricas aparte para plataformas compartidas y acortadores, e importancia y estabilidad de las características.

## Entradas

- `data/processed/20261112T025607Z/features_train.parquet`: 64,423 filas (32,212 phishing y 32,211 legítimas).
- `data/processed/20261112T025607Z/features_test.parquet`: 16,107 filas (8,053 y 8,054). Solo se usa en la evaluación final.
- Columna `group` (dominio registrado) para agrupar la validación cruzada.

## Método aprobado

Aprobado por Andrés el 2026-11-12.

| Decisión | Propuesta | Sustento |
|---|---|---|
| Entradas del meta-modelo | Probabilidades de cada modelo base predichas por pliegue (fuera de pliegue) sobre el entrenamiento | P15 (super learner por pliegues); P27 (regresión logística como meta-modelo) |
| Validación cruzada | 5 pliegues, estratificados y agrupados por dominio, con la misma semilla | 5 pliegues: P09, P07. Agrupación por dominio: decisión propia heredada de la Fase 1; el corpus no la trata |
| Ajuste de hiperparámetros | Búsqueda aleatoria con los rangos de P09, evaluada con F1 en la validación cruzada | P09 (RandomizedSearch y rangos); P07, P17, P20 y P24 usan búsqueda en rejilla |
| Umbral | Elegido en la validación cruzada, el de mayor F1 promedio; nunca con la prueba | P17 |
| Calibración | No se aplica | El corpus no la trata |
| Formato del artefacto | `joblib`, un archivo por versión, con su `report.json` | Decisión menor de implementación |

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
