# sereno-model

Pipeline de entrenamiento fuera de línea del ensemble Stacking de Sereno: Random Forest, XGBoost y LightGBM como modelos base y regresión logística como meta-modelo.

Corresponde a la "Estación de entrenamiento" del despliegue del ADD. Este repositorio no sirve predicciones: publica un artefacto versionado que el servidor de inferencia (`sereno-backend`) carga.

## Estado

Conjunto de datos limpio, partición fija y matrices de características (vector versión 2, 24 características) listos; aún sin modelo real. Fases 1 (datos) y 2 (características) cerradas; sigue la Fase 3, modelos. Instalación en `docs/00-entorno.md`.

## Documentación

- `docs/00-trazabilidad-arquitectura.md`: lo que el ADD ya decidió y lo que queda abierto.
- `docs/decisiones.md`: decisiones del modelo, con su motivo.
- `docs/NN-<fase>.md`: bitácora de cada fase, creada al iniciarla.

## Carpetas

| Carpeta | Contenido | En Git |
|---|---|---|
| `docs/` | Bitácora y decisiones | Sí |
| `data/` | Descargas originales y datos procesados (no van a Git) y `manifest.json` con fuente, fecha y hash (sí va) | Parcial |
| `notebooks/` | Exploración; los resultados oficiales salen de scripts | Sí |
| `src/sereno_model/` | Extracción de características, entrenamiento y evaluación | Sí |
| `tests/` | Pruebas | Sí |
| `artifacts/` | Modelos entrenados (no van a Git) y sus reportes `report.json` (sí van) | Parcial |
