# AGENTS.md: sereno-model

Pipeline de entrenamiento fuera de línea del ensemble Stacking de la tesis. Complementa el `AGENTS.md` de la carpeta de la tesis, que sigue vigente.

## Fuentes que mandan

1. El ADD OE2 y los diagramas C4. Resumen y abiertos en `docs/00-trazabilidad-arquitectura.md`.
2. `docs/decisiones.md`: decisiones ya tomadas para el modelo.
3. `docs/NN-<fase>.md`: estado y evidencia de cada fase.

Si el código y estas fuentes chocan, se detiene el trabajo y se avisa. No se cambia el diseño del ADD sin pedido de Andrés.

## Reglas

- Leer las fuentes anteriores antes de cualquier tarea.
- Entrenamiento oficial: solo en el equipo de Andrés y en CPU.
- Claude escribe el código. Una fase se aprueba solo después de que Andrés la explique con sus palabras.
- Todo lo que se afirme en la tesis debe poder rastrearse a un script, un dato con hash y un reporte guardado.
- Código, identificadores y comentarios en inglés. Documentación en español.
- Commits convencionales y sin atribución de IA. Solo se hace commit cuando Andrés lo pide.
- Nunca poner datasets ni modelos en Git.

## Skills del proyecto

| Skill | Cuándo |
|---|---|
| `fase-modelo` | Al abrir, avanzar o cerrar una fase |
| `experimento-reproducible` | Antes de cualquier corrida cuyo resultado se pueda reportar |
| `control-fuga-datos` | Antes de partir datos o de aceptar una métrica |

## Entorno

Ver `docs/00-entorno.md`.
