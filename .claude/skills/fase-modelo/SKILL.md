---
name: fase-modelo
description: "Trigger: abrir, avanzar o cerrar una fase del modelo, datos, features, Stacking, umbrales, integración. Controla la fase contra el ADD y documenta."
license: Proprietary
metadata:
  author: "andres-torres"
  version: "1.0"
---

## Activation Contract

Cargar al abrir, retomar o cerrar una fase de `sereno-model`.

## Hard Rules

- No empezar sin leer `docs/00-trazabilidad-arquitectura.md`, `docs/decisiones.md` y el documento de la fase.
- No contradecir el ADD. Si una tarea choca con él, detenerse y preguntar a Andrés.
- No inventar datos, citas ni resultados. Marcar lo no verificado como pendiente.
- No cerrar una fase sin la explicación de Andrés con sus palabras.
- No avanzar a la siguiente fase con una decisión abierta que dependa de esta.

## Decision Gates

| Situación | Acción |
|---|---|
| Decisión ya tomada en el ADD | Aplicarla y citar su origen |
| Decisión técnica menor | Decidir, registrar y avisar |
| Cambio de alcance o del diseño | Preguntar a Andrés |
| Explicación de Andrés incompleta o errónea | Señalar el vacío con evidencia y mantener la fase abierta |

## Execution Steps

1. Crear o abrir `docs/NN-<fase>.md` con objetivo, requisitos del ADD, entradas y criterio de salida.
2. Explicar el concepto a Andrés antes de escribir código.
3. Ejecutar la fase y registrar qué se hizo, por qué, alternativas descartadas y resultados.
4. Registrar las decisiones en `docs/decisiones.md`.
5. Pedir a Andrés que explique la fase y anotar el resultado de esa revisión.
6. Actualizar `docs/00-trazabilidad-arquitectura.md` y la fase actual en `README.md`.

## Output Contract

Reportar la fase, su estado, la evidencia, lo pendiente y la siguiente acción.

## References

- `../../../docs/00-trazabilidad-arquitectura.md`
- `../../../docs/decisiones.md`
