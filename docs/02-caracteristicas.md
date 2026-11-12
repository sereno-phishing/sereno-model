# Fase 2: características

Estado: abierta el 2026-11-11.

## Objetivo

Definir el vector de características de una URL e implementar su extracción en un solo paquete, que use el entrenamiento y que el servidor de inferencia pueda ejecutar sin cambios. La fase entrega además las matrices de características de entrenamiento y de prueba, con hash, listas para la Fase 3.

## Requisitos del ADD

| Requisito | Contenido | Origen |
|---|---|---|
| Entrada | Características léxicas y de host, solo intrínsecas de la URL; sin DOM ni HTML | Tabla 10, RES-01 |
| Vector | Definición idéntica en entrenamiento e inferencia | RF-04 |
| Ubicación | Paquete de `sereno-model`, ejecutado dentro de `sereno-backend` | `00-trazabilidad-arquitectura.md` |
| Privacidad | La URL completa se procesa solo en memoria durante la extracción | RNF-05 |
| Latencia | Promedio <= 200 ms de extremo a extremo; la extracción consume parte de ese margen | RNF-01 |
| Reproducibilidad | Misma partición y semilla entre versiones | Tablas 5 y 12 |

## Decisiones heredadas de la Fase 1

- El esquema y el «www.» quedan fuera del vector, salvo que el control de sesgo de fuente muestre que no delatan la fuente.
- El «www.» se quita antes de calcular la longitud y los subdominios.
- Si el control de sesgo de fuente falla, se agrega Common Crawl como fuente de legítimas con ruta, lo que cambia el Charter y el ADD.
- Las métricas se reportarán también por separado para plataformas compartidas y acortadores.

## Entradas

- Partición fija de la corrida `20261111T222736Z`, con hash `d7121875adbdf535a121c23afc4958f3e4b0b35e5b76a6811019948a55f89edc` y semilla 20261111.
- Normalización de `src/sereno_model/data/normalize.py`, la del criterio de «URL equivalente» del ADD (sección 3.1).

## Pendientes

| Tema | Estado |
|---|---|
| Características de host: solo desde el texto de la URL, o también con DNS o WHOIS | Decidido el 2026-11-11: solo desde el texto; ver `decisiones.md` |
| Lista de características del vector, con nombre y definición | Aprobada por Andrés el 2026-11-11; ver «Vector base» |
| Características que dependen de una lista (palabras sospechosas, acortadores, marcas, TLD sospechoso) | Fuera del vector base; se agregan después como un paso medido aparte |
| Extracción en un paquete único, con pruebas | Hecho el 2026-11-11; ver el registro |
| Control de sesgo de fuente sobre el vector | Ejecutado el 2026-11-11, corrida `20261112T024408Z`: falla con umbrales propios; falta sustento del corpus y la decisión de Andrés |
| Matrices de entrenamiento y de prueba con hash y reporte | Hecho el 2026-11-11; corrida `20261112T023011Z` |

## Vector base

Versión 2 del vector. Andrés aprobó la versión 1 el 2026-11-11 y, el mismo día, el cambio que da la versión 2: las barras con que termina la URL no se miden. Todas las características se calculan desde el texto de la URL ya normalizada con el criterio de «URL equivalente» del ADD.

El «texto medido» es la URL normalizada sin el esquema ni «://», sin el prefijo «www.» del host y sin las barras con que termina la URL. Así, `http://www.ejemplo.com/a` y `https://ejemplo.com/a` dan el mismo vector. El «host» es el host sin el prefijo «www.».

| N.º | Nombre | Definición | Papers del corpus (aprox.) |
|---|---|---|---|
| 1 | `url_length` | Caracteres del texto medido | 22 |
| 2 | `dot_count` | Puntos en el texto medido | 9 |
| 3 | `has_at` | 1 si el texto medido contiene «@» | 11 |
| 4 | `host_length` | Caracteres del host | 11 |
| 5 | `subdomain_count` | Etiquetas del host a la izquierda del dominio registrado; 0 si el host es una IP | 13 |
| 6 | `host_hyphen_count` | Guiones en el host | 13 |
| 7 | `host_is_ip` | 1 si el host es una dirección IP | 14 |
| 8 | `has_port` | 1 si la URL trae un puerto explícito distinto de 80 y de 443 | 4 |
| 9 | `tld_length` | Caracteres del sufijo público del host; 0 si es una IP o no tiene sufijo | 8 |
| 10 | `path_length` | Caracteres de la ruta | 9 |
| 11 | `path_depth` | Segmentos no vacíos de la ruta | 9 |
| 12 | `has_double_slash` | 1 si aparece «//» en la ruta o en la consulta | 7 |
| 13 | `query_length` | Caracteres de la consulta; 0 si no hay | 9 |
| 14 | `query_param_count` | Parámetros de la consulta, separados por «&» | 9 |
| 15 | `digit_count` | Dígitos en el texto medido | 10 |
| 16 | `digit_ratio` | `digit_count` entre `url_length` | 10 |
| 17 | `question_mark_count` | «?» en el texto medido | 13 |
| 18 | `equals_count` | «=» en el texto medido | 13 |
| 19 | `ampersand_count` | «&» en el texto medido | 13 |
| 20 | `percent_count` | «%» en el texto medido | 13 |
| 21 | `underscore_count` | «_» en el texto medido | 13 |
| 22 | `tilde_count` | «~» en el texto medido | 13 |
| 23 | `slash_count` | «/» en el texto medido | 13 |
| 24 | `url_entropy` | Entropía de Shannon, en bits, de los caracteres del texto medido | 5 |

El dominio registrado y el sufijo público salen de `tldextract` con su lista incluida, sin red y solo con sufijos ICANN, igual que en la partición de la Fase 1.

El conteo de «#» no entra: la normalización quita el fragmento, así que valdría siempre cero.

Casos límite, decididos al implementar:

| Caso | Regla |
|---|---|
| «www.» | Se quitan todos los «www.» iniciales mientras quede algo después; `www2.` se conserva. Quitar solo el primero dejaba que el prefijo cambiara el vector de un host como `www.www.ejemplo.com` |
| Puerto | 80 y 443 cuentan como puerto por defecto con cualquier esquema y no entran al texto medido. La normalización solo quita el del esquema propio, así que sin esta regla `http://host:443/` y `https://host:443/` darían vectores distintos y el esquema llegaría al vector |
| IP | Cuenta como IP un IPv4 decimal de cuatro partes o un IPv6 entre corchetes. Las formas entera, hexadecimal, octal o abreviada no cuentan |
| Host sin dominio registrado | IP, una sola etiqueta o sufijo desconocido: `subdomain_count` y `tld_length` valen 0. Si el host es un sufijo público, `tld_length` es su longitud |
| Sufijos | Solo ICANN: `victim.pages.dev` tiene un subdominio y sufijo `dev`. Las etiquetas vacías no cuentan |
| Usuario y contraseña | Quedan en el texto medido, así que cuentan para `has_at`, la longitud, los símbolos y la entropía; no forman parte del host |
| Ruta | Las barras con que termina la URL no se miden: `/a/b/` cuenta como `/a/b` y una `/` sola como ruta vacía (Ghalechyan et al., 2024; P30). La barra que precede a una consulta se conserva. Ruta vacía: longitud 0 y profundidad 0 |
| Consulta | Ausente o vacía («?» solo): longitud 0 y 0 parámetros. El «?» no cuenta en la longitud. Los parámetros son los tramos no vacíos entre «&» |
| Dígitos | Solo ASCII, del 0 al 9 |
| Entropía | Sobre los caracteres (puntos de código) del texto medido |

Quedan fuera del vector base, además del esquema y el «www.», las características que dependen de una lista y las representaciones aprendidas del texto (embeddings, TF-IDF, BERT), porque el ADD pide un vector definido de características léxicas y de host.

## Criterio de salida

- Cada característica del vector tiene nombre, definición y origen documentados.
- La extracción vive en un solo paquete, es determinista y tiene pruebas; el servidor de inferencia puede importarla sin cambios.
- Las matrices de entrenamiento y de prueba salen de un script, sobre código con commit, con hash y reporte guardado.
- Los controles de `control-fuga-datos` que aplican a las características están ejecutados y guardados, incluido el de sesgo de fuente.
- La pregunta abierta sobre las características de host está cerrada en `00-trazabilidad-arquitectura.md`.
- Andrés explicó la fase con sus palabras.

## Registro

### 2026-11-11: apertura

Se abre la fase. Antes de escribir código se explica el concepto a Andrés y se resuelve la pregunta abierta sobre las características de host.

### 2026-11-11: características de host

Decisión de Andrés: las características de host se calculan solo desde el texto de la URL, sin DNS ni WHOIS. Motivos y alternativas descartadas en `decisiones.md`.

### 2026-11-11: características en el corpus del TDP

Búsqueda dirigida en el texto extraído de los 40 papers (`P01.pdf.txt` a `P40.pdf.txt`, numeración de los PDF), con lectura de los pasajes sobre características y no de cada paper completo. Los conteos son aproximados, pueden variar en uno o dos y no incluyen tablas ni figuras en imagen. Varios papers reutilizan los mismos datasets públicos, así que parte de las coincidencias viene del dataset y no de una elección propia. Se comprobaron a mano tres de las citas (P21, P28 y P33).

Características calculables solo desde el texto de la URL:

| Característica | Grupo | Papers (aprox.) |
|---|---|---|
| Longitud de la URL | URL completa | 22 |
| IP como host | Host | 14 |
| Número de subdominios | Host | 13 |
| Guiones en el host o en la URL | Host | 13 |
| Conteo de símbolos (`? = & % _ ~ # /`) o su proporción | Caracteres | 13 |
| Longitud del host | Host | 11 |
| Presencia o conteo de «@» | URL completa | 11 |
| Palabras sospechosas (login, verify…) | Listas | 11 |
| Dígitos, conteo o proporción | Caracteres | 10 |
| Número de puntos | URL completa | 9 |
| Longitud y profundidad de la ruta | Ruta | 9 |
| Longitud de la consulta y número de parámetros | Consulta | 9 |
| TLD (categoría, longitud o lista de sospechosos) | Host | 8 |
| Servicio acortador | Listas | 8 |
| «//» después del esquema | Ruta | 7 |
| Marca en la URL | Listas | 7 |
| Entropía de Shannon | Caracteres | 5 |
| Token «http» o «https» dentro del host o la ruta | Host y ruta | 4 |
| Puerto explícito | Host | 4 |
| Letras, mayúsculas o vocales | Caracteres | 3 |
| Estadísticas de tokens | Tokens | 2 |

Catorce papers aprenden la representación del texto de la URL (embeddings, TF-IDF o BERT) en lugar de usar un vector definido a mano.

Sobre el «www.»: P19 y P30 lo quitan, y P33 lo excluye al contar puntos y subdominios, lo que coincide con la decisión de la Fase 1.

Fuera del vector por no ser calculables desde el texto: WHOIS y antigüedad del dominio, DNS, MX y certificado TLS, tráfico web y PageRank, contenido HTML o DOM, redirecciones seguidas en la red, geolocalización y listas negras.

Importancia reportada: cuando el dataset trae características de contenido o de certificado, estas quedan por encima de las léxicas (P07, P11, P20). Entre las léxicas, las más altas son la longitud de la URL, los guiones y los puntos (P07, P18, P20). P21 indica que la característica de marca no sirve en tiempo real por su costo de procesamiento.

Encabezados de los datasets: el de Dam Minh y Tran Cong (2026) trae solo características del texto, pero cuenta el «www.» como subdominio e incluye el esquema (`is_https`). El de Hannousse y Yahiouche (2021) mezcla características del texto con 24 de contenido y 7 de consultas externas. Ninguna columna calculada de esos archivos entra al vector: las características se calculan con el código de este repositorio.

Vacíos: P16, P21, P17, P01 y P25 no nombran en el texto su lista completa o su orden de importancia; P22, P37 y P38 no usan vectores de características de URL; el texto de P06, P07 y P20 quedó mezclado por las dos columnas. Ningún paper del corpus usa punycode: solo aparece como columna del dataset de Hannousse y Yahiouche (2021).

### 2026-11-11: vector base aprobado y extractor

Andrés aprobó el vector base de 24 características y dejó las que dependen de una lista para un paso medido aparte. Ver «Vector base» y `decisiones.md`.

Se escribieron, con pruebas primero, el extractor (`src/sereno_model/features/extract.py`) y el script que genera las matrices (`src/sereno_model/features/build.py`, `python -m sereno_model.features.build --data-dir data`). El extractor solo usa la biblioteca estándar y `tldextract`, no toca la red ni el disco y no registra la URL. El script comprueba el hash de la partición antes de usarla, guarda las 24 características junto a la etiqueta, el grupo, las fuentes y dos columnas de auditoría (`audit_scheme` y `audit_has_www`) que no forman parte del vector, y calcula las estadísticas por característica solo con entrenamiento.

Una prueba comprueba que `http://www.ejemplo.com/a?b=1` y `https://ejemplo.com/a?b=1` dan el mismo vector. Al revisar el extractor apareció una vía por la que el esquema llegaba al vector, el puerto, y se corrigió con la regla de la tabla de casos límite.

Pruebas al terminar esta entrada: 253 en verde (168 anteriores y 85 nuevas). Código sin commit.

Medición orientativa, con `--dry-run`, sin escribir nada y con cambios sin commit, por lo que no se cita: 80,530 URL en 2.1 s, unos 0.03 ms por URL en un hilo. La extracción no compromete los 200 ms de RNF-01.

### 2026-11-11: verificación independiente del extractor

Una revisión aparte, de solo lectura, comparó el código con la sección «Vector base» y probó URL adversas. Las 24 definiciones coinciden con el documento. Con la red bloqueada y las escrituras vigiladas, el extractor no intentó conexiones ni escribió en disco, y no importó pandas, numpy ni scikit-learn. Dos corridas con `--dry-run` sobre la partición real dieron los mismos hashes y no cambiaron `data/`.

Encontró dos defectos, corregidos con una prueba primero:

| Defecto | Corrección |
|---|---|
| El hash de la partición solo cubre la URL y su lado. Una etiqueta, un grupo o la marca de dominio sintético alterados en `split.parquet` pasaban sin error | El script compara la etiqueta y la marca con el conjunto ya verificado, vuelve a derivar el grupo desde la URL y se detiene si algo difiere |
| Con un host como `www.www.x.com`, el prefijo «www.» cambiaba el vector. Fue la única diferencia en 30,800 pares generados; la partición real no tiene hosts así | Se quitan todos los «www.» iniciales |

No encontró ninguna vía por la que el esquema llegue al vector.

Pruebas: 260 en verde (168 anteriores y 92 nuevas). Código sin commit.

Limitaciones que se declaran y no se corrigen en esta fase:

- Un host internacional escrito en Unicode y el mismo en punycode dan vectores distintos, porque la normalización del ADD no convierte uno en otro. La partición tiene 3 hosts en Unicode y 54 en punycode.
- Un host con punto final (`ejemplo.com.`) cuenta ese punto; hay 4 filas así.
- Un puerto con ceros a la izquierda (`:080`) cuenta como puerto explícito.
- El script solo avisa cuando el repositorio tiene cambios sin commit, igual que los de la Fase 1; esas corridas no se citan.

Pasa a la integración: `InvalidUrlError` incluye la URL en su mensaje. Si el servidor registra esa excepción, registraría la URL, lo que choca con RNF-05. Queda anotado en `00-trazabilidad-arquitectura.md`.

No se comprobó: que los valores decimales sean idénticos en otro equipo, ni que `sereno-backend` pueda importar el extractor.

### 2026-11-11: matrices oficiales

Corrida `20261112T023011Z` (hora UTC), sobre el commit `fe084db` y con el repositorio limpio. Entrada: partición `20261111T222736Z`, con su hash comprobado. Vector versión 1, 24 características. Reporte en `data/processed/20261112T023011Z/features_report.json`.

| | URL | Phishing | Legítimas | Hash del contenido |
|---|---|---|---|---|
| Entrenamiento | 64,423 | 32,212 | 32,211 | `3481745f29c5f9c379845bd749ea257c50c6b00c5f18a3ff448f99d087f0a068` |
| Prueba | 16,107 | 8,053 | 8,054 | `a6a0721edfabaf2a7e31cc49de2ad2c552217c9a81bc4438b447ea40428427f3` |

Dos corridas anteriores al commit, con `--dry-run`, dieron los mismos hashes. La extracción tardó 1.9 s para las 80,530 URL, unos 0.02 ms por URL en un hilo.

Estadísticas del reporte, calculadas solo con entrenamiento. La prueba no se resume.

| Característica | Mediana, phishing | Mediana, legítimas | Distinta de cero, phishing | Distinta de cero, legítimas |
|---|---|---|---|---|
| `url_length` | 32 | 21 | 100 % | 100 % |
| `host_length` | 20 | 13 | 100 % | 100 % |
| `dot_count` | 2 | 1 | 100 % | 100 % |
| `url_entropy` | 4.04 | 3.67 | 100 % | 100 % |
| `subdomain_count` | 1 | 0 | 59.0 % | 16.8 % |
| `digit_count` | 1 | 0 | 59.2 % | 11.2 % |
| `host_hyphen_count` | 0 | 0 | 35.4 % | 10.9 % |
| `query_length` | 0 | 0 | 14.9 % | 5.4 % |
| `path_length` | 7 | 5 | 90.8 % | 90.8 % |
| `has_at` | 0 | 0 | 1.2 % | 0.02 % |
| `has_double_slash` | 0 | 0 | 2.4 % | 0.1 % |
| `host_is_ip` | 0 | 0 | 0.3 % | 0.01 % |
| `has_port` | 0 | 0 | 0.05 % | 0.05 % |

Estas diferencias todavía no se interpretan como señal de phishing: parte puede venir de la fuente de cada clase. Eso lo decide el control de sesgo de fuente.

### 2026-11-11: diseño del control de sesgo de fuente

El diseño y sus umbrales se fijan aquí antes de correr el control, para no ajustarlos a la vista del resultado.

Pregunta: con las 24 características y nada más, ¿se puede adivinar el esquema de la URL, si traía «www.» o de qué fuente vino? Si se puede, un modelo de clases podría estar aprendiendo la fuente en lugar del phishing.

- Datos: solo la matriz de entrenamiento de la corrida `20261112T023011Z`. La de prueba no se abre.
- Estratos: cada medición se hace dentro de una clase y de una forma (con ruta o sin ruta), para que ni la clase ni la ruta expliquen el resultado. Son cuatro estratos.
- Objetivos por estrato: el esquema (https o http), el «www.» y cada fuente (si la URL figura en esa fuente).
- Sonda: un Random Forest de 200 árboles con al menos 5 URL por hoja, igual para todos los objetivos, evaluado con validación cruzada de 5 partes agrupada por dominio registrado y semilla 20261111. Se reporta el AUC sobre las predicciones hechas fuera de cada parte. Es una sonda de diagnóstico, no un modelo de la tesis.
- Por característica: la separación que logra cada una por sí sola frente a cada objetivo, para nombrar cuál delata la fuente.
- Umbrales: AUC menor de 0.65, pasa; entre 0.65 y 0.80, se revisa; mayor de 0.80, falla. Un objetivo con menos de 100 URL en su lado menor, o con menos de 5 dominios en algún lado, no se evalúa y se declara.

Un AUC de 0.5 equivale a adivinar al azar y uno de 1.0 a acertar siempre.

Qué se hace con un fallo lo decide Andrés. La decisión de la Fase 1 prevé agregar Common Crawl como fuente de legítimas con ruta, lo que cambia el Charter y el ADD.

### 2026-11-11: resultado del control de sesgo de fuente

Corrida `20261112T024408Z` (hora UTC), sobre el commit `f8ca39f` y con el repositorio limpio, con semilla 20261111 y un hilo. Entrada: matriz de entrenamiento de la corrida `20261112T023011Z`, con su hash comprobado. Reporte en `data/processed/20261112T024408Z/source_bias_report.json`. Es una corrida de exploración con validación cruzada sobre entrenamiento: no es un resultado del modelo.

| Estrato | Objetivo | URL | Positivas | AUC | Resultado | Característica que más separa |
|---|---|---|---|---|---|---|
| Legítimas sin ruta | Esquema | 10,970 | 8,292 | 0.656 | Se revisa | `path_length` 0.620 |
| Legítimas sin ruta | «www.» | 10,970 | 361 | 0.616 | Pasa | `subdomain_count` 0.576 |
| Legítimas sin ruta | Fuente: Hannousse | 10,970 | 247 | 0.781 | Se revisa | `url_entropy` 0.653 |
| Legítimas sin ruta | Fuente: Majestic Million | 10,970 | 906 | 0.702 | Se revisa | `dot_count` 0.648 |
| Legítimas sin ruta | Fuente: Dam Minh | 10,970 | 9,818 | 0.667 | Se revisa | `dot_count` 0.593 |
| Legítimas con ruta | Esquema | 21,241 | 6,914 | 0.702 | Se revisa | `path_length` 0.600 |
| Legítimas con ruta | «www.» | 21,241 | 16,956 | 0.780 | Se revisa | `subdomain_count` 0.715 |
| Legítimas con ruta | Fuente: Hannousse | 21,241 | 2,784 | 0.912 | Falla | `url_length` 0.776 |
| Legítimas con ruta | Fuente: Dam Minh | 21,241 | 18,458 | 0.913 | Falla | `url_length` 0.776 |
| Phishing sin ruta | Esquema | 11,018 | 8,360 | 0.818 | Falla | `path_length` 0.759 |
| Phishing sin ruta | «www.» | 11,018 | 360 | 0.816 | Falla | `subdomain_count` 0.741 |
| Phishing sin ruta | Fuente: Hannousse | 11,018 | 675 | 0.709 | Se revisa | `dot_count` 0.592 |
| Phishing sin ruta | Fuente: PhishTank | 11,018 | 7,541 | 0.644 | Pasa | `host_hyphen_count` 0.573 |
| Phishing sin ruta | Fuente: Dam Minh | 11,018 | 3,233 | 0.658 | Se revisa | `host_hyphen_count` 0.583 |
| Phishing con ruta | Esquema | 21,194 | 17,708 | 0.802 | Falla | `slash_count` 0.686 |
| Phishing con ruta | «www.» | 21,194 | 1,815 | 0.869 | Falla | `subdomain_count` 0.697 |
| Phishing con ruta | Fuente: Hannousse | 21,194 | 3,771 | 0.826 | Falla | `slash_count` 0.712 |
| Phishing con ruta | Fuente: PhishTank | 21,194 | 13,781 | 0.769 | Se revisa | `slash_count` 0.631 |
| Phishing con ruta | Fuente: Dam Minh | 21,194 | 5,554 | 0.619 | Pasa | `url_length` 0.564 |

Con los umbrales fijados antes de la corrida, los tres controles fallan: de 19 mediciones, 3 pasan, 9 se revisan y 7 fallan. El vector guarda información sobre el esquema, el «www.» y la fuente, aunque ninguno de los tres sea una característica.

Lo que se observa, sin interpretar más de lo medido:

- En las URL sin ruta, lo que más delata el esquema es `path_length`. En ese estrato la ruta solo puede ser vacía o «/», así que la diferencia es si la URL termina en barra o no. Eso depende de cómo escribe cada fuente sus URL y no del phishing.
- Las dos fuentes de legítimas con ruta se distinguen entre sí con un AUC de 0.91, sobre todo por la longitud de la URL y de la ruta. Son dos colecciones armadas con criterios distintos.
- El «www.» se recupera por el número de subdominios y de puntos, aunque el prefijo se quita antes de contar.

Límite de este control: la sonda y los umbrales de 0.65 y 0.80 son un diseño propio de este repositorio. No salen de los papers del corpus, así que el veredicto de «falla» no tiene todavía sustento en la literatura. El control mide que la fuente se puede identificar; no mide cuánto de la separación entre phishing y legítimas se debe a la fuente.

### 2026-11-11: el sesgo entre fuentes en el corpus del TDP

Indicación de Andrés: las decisiones técnicas se sustentan primero en los 40 papers del corpus y, si no alcanzan, en fuentes indexadas con PDF público.

Búsqueda dirigida en el texto extraído de los 40 papers, con lectura de los pasajes y no de cada paper completo. Se comprobaron a mano las citas de P19, P20 y P30.

| Tema | Qué hace el corpus | Papers |
|---|---|---|
| Forma de la URL antes de extraer | Quitan el «www.» y las barras finales; P02 quita además el esquema | P19 (Ghalechyan et al., 2024), P30, P02 |
| Desbalance de forma entre clases | P19 exige que el conjunto no tenga desbalance de longitud ni de profundidad de URL entre clases y, si lo tiene, pide ponderar los ejemplos. No da procedimiento ni cifras | P19 |
| Sesgo por mezclar fuentes | Solo P19 lo relaciona con la forma de la URL. P38 lo declara como limitación, en datasets de red | P19, P38 |
| Validación con otro dataset | P20 entrena con un dataset y prueba con otro: pasa de 94.46 % a 80 % de exactitud. P29 prueba con un conjunto no visto que solo tiene phishing: 95.5 % | P20, P29 |
| Validación con datos más nuevos | P19 prueba con phishing recogido después del entrenamiento | P19 |
| Duplicados | Los quitan antes de partir | P02, P10, P14, P17, P26, P28, P32, P34 |

No aparece en el corpus:

- Una medición que prediga la fuente de una URL, como la sonda de este repositorio, ni umbrales para ella.
- La partición agrupada por dominio. Es una decisión de la Fase 1 que va más allá de lo que hacen los papers.
- Common Crawl elegido para corregir la forma de las URL. Cuatro papers lo usan como fuente de legítimas (P02, P04, P05, P29) sin dar ese motivo.
- Quitar una característica por ser un artefacto de la fuente.

Consecuencia: el veredicto de «falla» del control anterior se apoya en umbrales propios. Lo que el corpus sí respalda es quitar las barras finales antes de extraer (P19, P30), que atiende la señal de `path_length` en las URL sin ruta, y medir el efecto de la fuente probando con otro dataset (P20).

### 2026-11-11: versión 2 del vector, sin la barra final

Decisión de Andrés: las barras con que termina la URL no se miden, como hacen Ghalechyan et al. (2024) y P30, y se vuelven a correr las matrices y el control. Ver `decisiones.md`.

Propuesto y sin confirmar por Andrés: reportar el control como diagnóstico y limitación declarada, sin umbrales de pasa o falla; medir el efecto de la fuente en la Fase 3 probando con otra fuente, como P20; y no agregar Common Crawl por ahora.

El cambio afecta a `url_length`, `path_length`, `slash_count`, `has_double_slash` y `url_entropy`. El vector pasa a la versión 2 y las matrices de la versión 1 (corrida `20261112T023011Z`) quedan reemplazadas.

Pendiente: matrices y control de la versión 2 sobre código con commit.
