---
name: experimento-reproducible
description: "Trigger: entrenar, evaluar, experimento, hiperparámetros, métricas, artefacto del modelo. Exige semilla, partición fija, hash de datos y reporte guardado."
license: Proprietary
metadata:
  author: "andres-torres"
  version: "1.0"
---

## Activation Contract

Cargar antes de una corrida cuyo resultado se pueda reportar o publicar.

## Hard Rules

- Ejecutar desde scripts versionados, no desde celdas de notebook.
- Fijar la semilla y usar la misma partición entre versiones del modelo.
- Registrar: commit, comando, semilla, hash de los datos, hash de la partición, versiones de librerías, equipo, parámetros, métricas y duración.
- No reportar un resultado de una corrida con cambios sin commit.
- No usar el conjunto de prueba para elegir modelo, features, hiperparámetros ni umbrales.
- El entrenamiento oficial corre en CPU en el equipo de Andrés.
- Nunca sobrescribir un reporte ni un artefacto ya publicado.

## Decision Gates

| Corrida | Conjunto permitido | Se puede citar en la tesis |
|---|---|---|
| Exploración | Entrenamiento y validación cruzada | No |
| Selección de modelo o hiperparámetros | Validación cruzada | Solo como proceso |
| Evaluación final | Prueba ciega, una vez por versión | Sí |

## Execution Steps

1. Confirmar que el repositorio está limpio y que los hashes de datos coinciden.
2. Declarar el tipo de corrida y su hipótesis.
3. Ejecutar y guardar el reporte en `artifacts/<version>/`.
4. Resumir el resultado en el documento de la fase.
5. Si es una evaluación final, cargar antes `control-fuga-datos`.

## Output Contract

Reportar tipo de corrida, comando, ruta del reporte, métricas e incidencias.

## References

- `../../../docs/00-entorno.md`
