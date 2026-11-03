---
name: control-fuga-datos
description: "Trigger: partir datos, remuestreo, split, F1 alto, aceptar métrica, fuga de datos, sesgo. Busca fugas y sesgos antes de aceptar un resultado."
license: Proprietary
metadata:
  author: "andres-torres"
  version: "1.0"
---

## Activation Contract

Cargar al construir la partición, al remuestrear y antes de aceptar cualquier métrica.

## Hard Rules

- Tratar un F1 alto como sospechoso hasta completar este control.
- No aceptar una métrica con un control fallido o sin ejecutar.
- Ajustar remuestreo, escalado y selección de features solo con datos de entrenamiento.
- Entrenar el meta-modelo solo con predicciones out-of-fold.
- Usar solo features calculables en inferencia con la URL recibida.

## Decision Gates

| Control | Falla si |
|---|---|
| Duplicados | La misma URL normalizada aparece en entrenamiento y prueba |
| Dominios | Un mismo dominio aparece en entrenamiento y prueba, salvo excepción documentada |
| Sesgo de forma | La clase se separa casi solo por rutas, esquema o longitud por venir de fuentes distintas |
| Sesgo de fuente | Una feature identifica la fuente en lugar del phishing |
| Temporal | No se reporta el rango de fechas, o las clases provienen de épocas muy distintas sin justificarlo |
| Importancia | Una sola feature domina y no tiene explicación razonable |
| Estabilidad | La prueba supera con claridad a la validación cruzada |

## Execution Steps

1. Ejecutar cada control y guardar su resultado con el reporte.
2. Comparar con una línea base simple y con una prueba por dominios no vistos.
3. Registrar los hallazgos en el documento de la fase.
4. Corregir, volver a correr y documentar la causa si un control falla.

## Output Contract

Devolver una tabla con control, resultado, evidencia y acción.

## References

- `../experimento-reproducible/SKILL.md`
