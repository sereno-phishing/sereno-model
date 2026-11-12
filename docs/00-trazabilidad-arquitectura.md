# Trazabilidad con la arquitectura

Fuente: ADD OE2 (secciones del componente de IA, Tablas 3, 5, 10 y 11, Figura 25) y diagramas C4 de componentes. Si este archivo y el ADD chocan, manda el ADD.

## Ya decidido

| Tema | Decisión | Origen |
|---|---|---|
| Problema | Clasificación supervisada binaria: phishing o legítima | Tabla 10 |
| Modelos | Random Forest, XGBoost y LightGBM; meta-modelo regresión logística; scikit-learn | Tabla 10 |
| Entrada | Vector de características léxicas y de host, solo intrínsecas de la URL; sin DOM ni HTML | Tabla 10, RES-01 |
| Vector | Definición idéntica en entrenamiento e inferencia | RF-04 |
| Salida | Veredicto, puntaje entre 0 y 1 y versión del modelo | RF-05, Tabla 10 |
| Datos | Mendeley Data, PhishTank y Majestic Million; solo fuentes abiertas | RES-03, sección de datos |
| Preparación | Limpieza, remuestreo por desbalance y partición fija de entrenamiento y prueba | Figura 25 |
| Reproducibilidad | Misma partición y semilla entre versiones | Tabla 12, Tabla 5 |
| Entrenamiento | Fuera de línea, en equipos propios, sin GPU | Tabla 3 |
| Publicación | Un artefacto versionado por versión, con reversión, en el sistema de archivos del servidor | RF-11, Tabla 6 |
| Evaluación | Conjunto de prueba ciego: F1 >= 0.96; se reportan precisión, recall y exactitud | Tabla 11, RC-01 |
| Latencia | Promedio <= 200 ms de extremo a extremo; HIT y MISS por separado | RNF-01 |
| Privacidad | La URL completa se procesa solo en memoria durante la extracción | RNF-05 |
| Ejecución | En el servidor y solo cuando el caché no resuelve la consulta | RF-02, RNF-02 |

## Ubicación de los componentes

| Componente | Vive en | Motivo |
|---|---|---|
| Pipeline de entrenamiento | `sereno-model` | Estación de entrenamiento, fuera de línea |
| Extracción de Características | Paquete de `sereno-model`, ejecutado dentro de `sereno-backend` | Componente del servidor con una sola definición del vector (RF-04) |
| Modelo Ensemble (artefacto) | Se produce aquí y se carga en `sereno-backend` | Tabla 3 y RF-11 |

## Abierto

| Tema | Pendiente | Fase |
|---|---|---|
| Clave del caché | Andrés decidió usar la URL normalizada; falta corregir el backend, que usa el dominio (I-46) | Integración |
| Forma del veredicto | El puerto `Classifier` del backend devuelve tres estados y el ADD define dos (I-49) | Integración |
| URL en los errores | `InvalidUrlError` incluye la URL en su mensaje; el servidor no debe registrarlo tal cual (RNF-05) | Integración |

## Decidido en la Fase 1, compatible con el ADD

- Entrenamiento y prueba se separan por dominio registrado, para que un mismo dominio no aparezca en ambos.
- El balance de clases sale del armado del conjunto: tope de 500 URL de phishing por dominio y muestreo de las legítimas por estratos de forma. Ocurre antes de la partición, como dice la sección 8.2, y no se aplica otro remuestreo en entrenamiento.
- El esquema y el «www.» quedan fuera del vector de características, salvo que el control de sesgo de fuente muestre que no delatan la fuente.

## Decidido en la Fase 2, compatible con el ADD

- Las características de host se calculan solo desde el texto de la URL, sin DNS, WHOIS ni otros servicios externos. El ADD pide características intrínsecas (Tabla 10, RES-01) y no menciona esas consultas.
