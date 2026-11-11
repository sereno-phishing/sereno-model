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
| Combinar las URLs legítimas sin sesgo de forma | Decidido el 2026-11-11; el sesgo con ruta se verifica en la Fase 2 |
| Limpieza, duplicados y etiquetas contradictorias | Hecho el 2026-11-11; ver el registro |
| Fechas de las URLs legítimas | Pendiente |
| Separar entrenamiento y prueba por dominio | Implementado el 2026-11-11; ver el registro |
| Remuestrear solo el entrenamiento | Por confirmar: las clases quedan parejas tras el armado |

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

### 2026-11-11: sesgo de esquema y de «www.»

Decisión de Andrés: las legítimas se muestrean por estratos para igualar la proporción con ruta y sin ruta de las de phishing, y una parte acotada del estrato sin ruta sale de Majestic Million.

Exploración sobre la corrida `20261111T215311Z`, hecha fuera de un script: orienta la fase y no se cita.

| Grupo | URL | https | Con «www.» | Longitud mediana |
|---|---|---|---|---|
| Legítimas sin ruta | 77,201 | 36.5 % | 73.8 % | 24 |
| Phishing sin ruta | 21,128 | 82.8 % | 2.1 % | 35 |
| Legítimas con ruta | 27,131 | 32.4 % | 79.9 % | 38 |
| Phishing con ruta | 45,977 | 90.3 % | 6.4 % | 43 |

El esquema y el prefijo «www.» separan las clases casi por sí solos, y eso viene de cómo y cuándo se recogió cada fuente, no del phishing. Igualar solo la proporción con ruta no lo corrige. Solo el 11.6 % de las legítimas sin ruta usa https sin «www.», la forma del 81.5 % de las de phishing sin ruta.

De los dominios de Majestic Million, 31,380 ya aparecen como host en las URL limpias y 391 son host de alguna URL de phishing. Entre las URL limpias, 21 hosts tienen URL de ambas clases.

Resuelto en el armado del conjunto, más abajo.

### 2026-11-11: armado del conjunto

`python -m sereno_model.data.assemble --data-dir data` conserva todo el phishing y muestrea las legítimas por ocho estratos de forma (con o sin ruta, esquema y «www.») para imitar la distribución del phishing. Majestic Million solo cubre lo que falte en los estratos sin ruta, con dominios del top 100,000 que no aparezcan ya como host. Con `--dry-run` compara proporciones sin escribir nada.

Corridas con cambios sin commit: orientan la fase y no se citan. La distancia es la variación total entre las distribuciones de forma de las dos clases; antes de muestrear vale 0.759 en total, 0.786 con ruta y 0.718 sin ruta.

| Legítimas por phishing | Legítimas | De Majestic | Distancia total | Con ruta | Sin ruta |
|---|---|---|---|---|---|
| 0.3 | 20,132 | 0 | 0.489 | 0.714 | 0.0001 |
| 0.4 | 26,843 | 0 | 0.514 | 0.750 | 0.0001 |
| 0.5 | 33,555 | 0 | 0.529 | 0.772 | 0.0001 |
| 0.59 | 39,592 | 1,160 | 0.538 | 0.786 | 0.0001 |

Sin ruta, el muestreo elimina la diferencia de forma. Con ruta casi no la cambia: 39,515 URL de phishing usan https sin «www.» y solo 2,005 legítimas tienen esa forma, así que el resto se completa con legítimas de otras formas. Majestic Million no ayuda ahí porque no trae rutas.

Hasta una proporción de 0.5 no hace falta ningún dominio de Majestic Million.

Decisión de Andrés: proporción de 0.59, y el sesgo de forma de las URL con ruta se trata en las características de la Fase 2. Ver `decisiones.md`.

Corrida `20261111T220928Z`, sobre el commit `985b8a5` y con el repositorio limpio, con proporción 0.59 y semilla 20261111: 106,697 URL, 67,105 de phishing y 39,592 legítimas, de las cuales 1,160 salen de Majestic Million. La distancia de forma queda en 0.0001 sin ruta y en 0.786 con ruta. Hash del contenido: `b1d5830a7ae19f5c22b236893730357db8c9b87f64d4e4aa13a3bad56dbb3cf6`. La tabla comparativa de arriba salió de corridas anteriores al commit.

### 2026-11-11: partición por dominio y concentración del phishing

`python -m sereno_model.data.split --data-dir data` parte el conjunto en entrenamiento y prueba, 80/20, agrupando por dominio registrado para que ningún dominio quede en los dos lados. El dominio se obtiene con `tldextract` 5.4.0 y la lista de sufijos que trae incluida, sin red y solo con sufijos ICANN: todos los sitios de una plataforma compartida forman un grupo. La proporción 80/20 es la de Sankaranarayanan et al. (2024) y Naseeb et al. (2025); el ADD no fija ninguna.

Corrida `20261111T221721Z`, con cambios sin commit: orienta la fase y no se cita.

| | URL | Phishing | Legítimas | Grupos |
|---|---|---|---|---|
| Entrenamiento | 85,358 | 53,684 | 31,674 | 35,927 |
| Prueba | 21,339 | 13,421 | 7,918 | 8,983 |

Controles de duplicados y de dominios: 0 URL y 0 grupos en ambos lados.

El phishing está concentrado en pocos dominios: los 10 mayores reúnen el 45.1 % y los 50 mayores el 64.0 %. Son plataformas de alojamiento y acortadores: `google.com` (6,332 URL, casi todas de `docs.google.com` y `sites.google.com`), `qrco.de`, `weebly.com`, `q-r.to`, `ead.me`, `web.app`, `firebaseapp.com`, `bit.ly`, `pages.dev` y `r2.dev`. Las legítimas no están concentradas: su mayor grupo tiene 178 URL.

Con esta semilla, `google.com` quedó entero en la prueba y es el 47 % de su phishing, y entrenamiento no tiene ninguna URL de ese dominio. La diferencia de forma sin ruta, que era 0.0001 en el conjunto completo, subió a 0.015 en entrenamiento y a 0.079 en prueba.

Decisión de Andrés: tope de 500 URL de phishing por dominio registrado, elegidas al azar con la semilla.

### 2026-11-11: conjunto y partición con el tope por dominio

El tope se aplica en `assemble`, antes de calcular las cuotas de legítimas. Recorta 16 dominios y deja 40,265 URL de phishing de las 67,105 limpias. Los dominios recortados son `google.com`, `qrco.de`, `weebly.com`, `q-r.to`, `ead.me`, `web.app`, `firebaseapp.com`, `bit.ly`, `pages.dev`, `r2.dev`, `adobe.com`, `campaign-archive.com`, `wixstudio.com`, `dropbox.com`, `blogspot.com` y `vercel.app`.

Con menos phishing, la proporción pasa a 1.0 legítimas por cada URL de phishing. Sigue el mismo criterio que antes: la mayor proporción sin faltantes, que usa casi todas las legítimas con ruta e incluye Majestic Million, sin pasar de clases parejas.

Corridas `20261111T222725Z` (conjunto) y `20261111T222736Z` (partición), sobre el commit `aaef0a8` y con el repositorio limpio, con semilla 20261111. Hash del conjunto: `b3a2e8994c11aeb0e94e8c306e206df664ffcd29eb04e7e15135d449837b5f9c`. Hash de la partición: `d7121875adbdf535a121c23afc4958f3e4b0b35e5b76a6811019948a55f89edc`. Dos corridas anteriores al commit dieron los mismos hashes.

| | URL | Phishing | Legítimas | De Majestic | Grupos |
|---|---|---|---|---|---|
| Conjunto | 80,530 | 40,265 | 40,265 | 1,113 | 45,633 |
| Entrenamiento | 64,423 | 32,212 | 32,211 | 906 | 36,505 |
| Prueba | 16,107 | 8,053 | 8,054 | 207 | 9,128 |

Controles de duplicados y de dominios: 0 URL y 0 grupos en ambos lados. El mayor dominio de phishing pesa el 1.6 % en entrenamiento y el 6.2 % en prueba; antes del tope, `google.com` era el 47 % del phishing de prueba. Los 16 dominios recortados suman el 18.6 % del phishing de entrenamiento y el 24.8 % del de prueba.

La distancia de forma entre clases queda en 0.003 sin ruta y 0.713 con ruta en entrenamiento, y en 0.012 sin ruta y 0.617 con ruta en prueba. La de las URL con ruta se trata en las características de la Fase 2.

Las clases quedan parejas, así que el muestreo de las legítimas ya cumple el papel del remuestreo por desbalance que pide el ADD, y ocurre antes de la partición, como indica su sección 8.2. Falta confirmar con Andrés si hace falta otro remuestreo en entrenamiento.

## Revisión de Andrés

Pendiente.
