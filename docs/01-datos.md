# Fase 1: datos

Estado: abierta el 2026-11-03.

## Objetivo

Obtener un conjunto de URLs etiquetadas (phishing o legítima), limpio, con hash y con una partición fija de entrenamiento y prueba, listo para la extracción de características de la Fase 2.

## Requisitos del ADD

| Requisito | Contenido | Origen |
|---|---|---|
| Problema | Clasificación supervisada binaria | Tabla 10 |
| Entrada | Solo características intrínsecas de la URL; sin DOM ni HTML | Tabla 10, RES-01 |
| Fuentes | Mendeley Data, PhishTank y Majestic Million; solo fuentes abiertas | RES-03, sección de datos |
| Preparación | Limpieza, remuestreo por desbalance y partición fija | Figura 25 |
| Reproducibilidad | Misma partición y semilla entre versiones | Tablas 5 y 12 |

## Entradas

- Descargas originales en `data/raw/`, cada una con hash, fecha y origen.

## Pendientes

| Tema | Estado |
|---|---|
| Revisar qué datasets usa la literatura y si cumplen RES-01 | Corpus revisado; literatura 2023-2026 incompleta |
| Selección de fuentes | Aprobada por Andrés; ver `decisiones.md` |
| OpenPhish | En pausa: falta confirmar si el proyecto puede ampararse en sus términos |
| Términos de PhishTank | Sin leer: la página remite a un EULA de Cisco que no se pudo abrir |
| Combinar las URLs legítimas sin sesgo de forma | Abierto |
| Limpieza, duplicados y etiquetas contradictorias | Hecho el 2026-11-11; ver el registro |
| Fechas de las URLs legítimas | Pendiente |
| Separar entrenamiento y prueba por dominio | Propuesta por confirmar |
| Remuestrear solo el entrenamiento | Propuesta por confirmar |

## Criterio de salida

- Cada fuente usada tiene hash, fecha y origen registrados.
- La partición está guardada y se reproduce con la misma semilla.
- La tabla de `control-fuga-datos` que aplica a los datos está ejecutada y guardada.
- I-47 está resuelto en lo que toca a UCI y Mendeley.
- Andrés explicó la fase con sus palabras.

## Registro

### 2026-11-03: apertura

Se abre la fase. Antes de descargar datos se revisan los datasets usados en la literatura.

### 2026-11-03: datasets en el corpus del TDP

Búsqueda dirigida en el texto ya extraído de los 40 papers, con lectura de los pasajes sobre datos, no de cada paper completo. Los conteos son aproximados: incluyen usos inferidos y no se revisaron figuras ni tablas en imagen. Solo cuentan los usos directos en experimentos propios. El corpus es el del TDP, no una muestra de 2023-2026.

| Dataset | Papers | Sin MDPI |
|---|---|---|
| PhishTank | 14 | 10 |
| Kaggle (varios datasets) | 12 | 10 |
| Mendeley (al menos tres datasets) | 8 | 6 |
| Alexa | 5 | 3 |
| UCI Phishing Websites | 4 | 2 |
| OpenPhish | 4 | 3 |
| Common Crawl | 4 | 2 |
| PhishStorm | 3 | 3 |
| ISCX-URL2016 | 2 | 2 |
| Majestic Million | 1 | 1 |
| PhiUSIIL | 1 | 1 |

Majestic Million aparece además en P29, en un pasaje que parece describir trabajos relacionados; no se cuenta como uso directo.

### 2026-11-03: páginas oficiales de los datasets

Revisión parcial. No se pudo leer el texto completo de ningún paper revisado por pares de 2023-2026, por lo que no hay tabla de frecuencias de ese periodo.

| Fuente | URLs crudas | Observación |
|---|---|---|
| UCI Phishing Websites | La página solo lista atributos calculados | 11,055 filas y 30 atributos; algunos nombres sugieren inspección de la página |
| PhiUSIIL (UCI) | Sí | 235,795 filas; CC BY 4.0; atributos de la URL y del código fuente |
| Mendeley, Hannousse (c2gw7fy2j4) | Sí | 11,430 filas, 50/50; CC BY 4.0; parte de los atributos usa contenido y servicios externos |
| Mendeley, URL-Phish (65z9twcx3r) | Sí | 116,600 filas, 86/14; CC BY 4.0; las legítimas vienen de registros institucionales |
| Mendeley, Vrbančič (72ptz43s9v) | No confirmado | 88,647 y 58,645 filas; 111 atributos |
| PhishTank | Sí | Solo phishing; clave solicitada y límite de descargas sin ella; registro no verificado |
| OpenPhish | Sí | Solo phishing; los términos restringen obras derivadas, redistribución y desarrollo de productos de detección; falta leer la página de uso académico |
| Majestic Million | No, dominios | Solo legítimas; CC BY 3.0 |
| Tranco | No, dominios | Solo legítimas; fuera de la lista del ADD |

Fuentes: <https://archive.ics.uci.edu/dataset/327/phishing+websites>, <https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset>, <https://data.mendeley.com/datasets/c2gw7fy2j4/3>, <https://data.mendeley.com/datasets/65z9twcx3r/2>, <https://data.mendeley.com/datasets/72ptz43s9v/1>, <https://phishtank.org/developer_info.php>, <https://openphish.com/terms.html>, <https://majestic.com/reports/majestic-million>, <https://tranco-list.eu/>.

### 2026-11-03: primera descarga

`python -m sereno_model.data.acquire` guarda cada fuente en una carpeta con fecha y registra hash, tamaño, fecha, licencia y cita en `data/manifest.json`. Los archivos de Mendeley se comprueban contra el SHA-256 que publica su API.

PhishTank se descargó sin clave porque el registro está cerrado. Sin clave se permiten pocas descargas al día, y la página no da la cifra. Sus URLs van del 2012-06-14 al 2026-11-04, con mediana en 2025-03-12. Las fechas de las demás fuentes quedan pendientes para el control temporal.

De Hannousse y Yahiouche (2021) solo se obtuvo `dataset_B_05_2020.csv`, el único archivo que lista la raíz de la API.

Perfil de forma, antes de limpiar. "Con ruta" significa que la URL tiene ruta o consulta; el resto es solo dominio.

| Fuente | Clase | Filas | Con ruta |
|---|---|---|---|
| PhishTank | Phishing | 50,320 | 71.6 % |
| Hannousse y Yahiouche (2021) | Phishing | 5,715 | 81.5 % |
| Dam Minh y Tran Cong (2026) | Phishing | 16,600 | 58.0 % |
| Hannousse y Yahiouche (2021) | Legítima | 5,715 | 61.5 % |
| Dam Minh y Tran Cong (2026) | Legítima | 100,000 | 23.9 % |
| Majestic Million | Legítima | 1,000,000 | 0 % |

Esto confirma el riesgo de sesgo de forma: mezclar las fuentes sin control dejaría casi todas las legítimas sin ruta. Las filas aún no están deduplicadas. Dam Minh y Tran Cong (2026) toman su phishing de PhishTank, así que puede haber solapamiento.

### 2026-11-11: documentos alineados con la selección

Andrés mantuvo la selección de fuentes del 2026-11-03. El Charter, la sección 8.2 y la Figura 25 del ADD, el Product Backlog, la Memoria, el Capstone y el paper nombran ahora Mendeley Data, PhishTank y Majestic Million. UCI Phishing Websites y OpenPhish ya no figuran como fuentes del proyecto.

### 2026-11-11: limpieza y duplicados

`python -m sereno_model.data.clean --data-dir data` comprueba el hash de cada descarga contra `data/manifest.json`, normaliza cada URL, descarta las inválidas, quita duplicados y aparta las URL con etiquetas contradictorias. La normalización es la del criterio de «URL equivalente» del ADD (sección 3.1), en `src/sereno_model/data/normalize.py`. La salida queda en `data/processed/<fecha>/`, con `clean_report.json`.

Corrida `20261111T215311Z`, sobre el commit `e703cf5` y con el repositorio limpio. Una corrida anterior del mismo día, hecha antes del commit, dio el mismo hash de contenido.

| Fuente | Filas leídas | Inválidas | Duplicadas en la fuente | Phishing | Legítimas |
|---|---|---|---|---|---|
| Hannousse y Yahiouche (2021) | 11,430 | 4 | 4 | 5,707 | 5,715 |
| Dam Minh y Tran Cong (2026) | 116,600 | 3 | 1,382 | 16,576 | 98,639 |
| PhishTank | 50,320 | 7 | 66 | 50,247 | 0 |

Las 14 inválidas tienen espacios o caracteres de control. Una URL presente en varias fuentes cuenta en cada una.

Entre fuentes se repiten 5,447 URL, todas con la misma etiqueta: 5,403 entre Dam Minh y Tran Cong (2026) y PhishTank, 22 entre Hannousse y Yahiouche (2021) y PhishTank, y 22 entre los dos datasets de Mendeley. No hay etiquetas contradictorias.

Quedan 171,437 URL únicas: 67,105 de phishing y 104,332 legítimas. Tienen ruta o consulta el 68.5 % de las de phishing y el 26.0 % de las legítimas, así que el sesgo de forma sigue presente después de limpiar.

En Dam Minh y Tran Cong (2026) la etiqueta 1 es phishing: 5,405 de sus 16,600 URL con etiqueta 1 están en PhishTank y ninguna de las 100,000 con etiqueta 0.

Majestic Million se guarda aparte como dominios (1,000,000, sin inválidos ni duplicados). No se convierten en URL: cómo se combinan con las legítimas sigue abierto.

Hash del contenido limpio: `95260c626a56f16123dce18f62773491126d6e9af1e74220e3177b810767ddbc`.

## Revisión de Andrés

Pendiente.
