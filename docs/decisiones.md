# Decisiones del modelo

Formato: fecha, decisión, motivo y alternativas descartadas.

## 2026-11-03: repositorio propio

`sereno-model` vive aparte de `sereno-backend`, al mismo nivel. El ADD separa la estación de entrenamiento del servidor de inferencia. Se descartó una carpeta dentro del backend porque mezclaría datasets, experimentos y dependencias de análisis con el servicio.

## 2026-11-03: equipo de entrenamiento

El entrenamiento oficial se ejecuta solo en el equipo de Andrés: Intel Core i7-12650H, 16 hilos, 31.6 GB de RAM, Windows y Python 3.12. Corre en CPU, como indica el ADD.

La GPU (RTX 3070 Laptop, 8 GB) no se usa en el entrenamiento oficial. Random Forest no la aprovecha y LightGBM en Windows solo admite la versión OpenCL, que acelera parte del proceso y por defecto usa precisión de 32 bits. Queda como apoyo opcional para explorar hiperparámetros con XGBoost, si se mide que hace falta.

## 2026-11-03: ubicación del entorno y los datos

El entorno virtual vive fuera de OneDrive, en `%USERPROFILE%\.venvs\sereno-model`, para no sincronizar miles de archivos. Las descargas originales quedan en `data/raw/`, dentro de OneDrive, porque los feeds de phishing no se pueden volver a descargar iguales y así tienen respaldo.

## 2026-11-03: fuentes de datos

Decisión de Andrés sobre la propuesta de la Fase 1. Se usan las fuentes del ADD que entregan URLs:

| Clase | Fuentes |
|---|---|
| Phishing | PhishTank y URLs de phishing de Mendeley (Hannousse y Yahiouche, 2021; Dam Minh y Tran Cong, 2026) |
| Legítima | URLs legítimas de esos mismos datasets de Mendeley y dominios de Majestic Million |

UCI Phishing Websites queda fuera del entrenamiento: su página solo lista atributos calculados y algunos parecen depender de la página, lo que choca con RES-01. El dataset de Vrbančič (2020) se descarta porque su descripción no menciona URLs crudas.

OpenPhish queda en pausa hasta aclarar sus términos (I-50). Common Crawl se usa solo si el sesgo de forma persiste, porque agregarlo implica cambiar el Charter y el ADD.

Se descartó elegir por número de menciones en los papers: varios de los más citados entregan atributos ya calculados, y Kaggle y Mendeley agrupan datasets distintos.

## 2026-11-11: fuentes en los documentos de la tesis

Andrés mantuvo la selección del 2026-11-03. El Charter, el ADD y los demás documentos de TP1 se alinearon a ella en lugar de reincorporar UCI Phishing Websites y OpenPhish.

## 2026-11-11: limpieza y duplicados

Las URL se comparan después de aplicar la normalización del criterio de «URL equivalente» del ADD (sección 3.1): esquema y host en minúsculas, sin puerto por defecto, sin fragmento y con los caracteres no reservados decodificados. Se conservan como distintos http y https, el prefijo «www.», la barra final y el orden de los parámetros. Así el entrenamiento y el caché usan la misma definición. Se descartó una normalización más agresiva porque el ADD la excluye.

Una URL con etiquetas contradictorias entre fuentes se excluye y se registra aparte; no se elige una etiqueta. Una URL repetida con la misma etiqueta se conserva una vez, con todas sus fuentes.

Son inválidas las URL vacías, sin host, con un esquema distinto de http o https, o con espacios o caracteres de control. De PhishTank solo entran las filas verificadas.

Los dominios de Majestic Million no se convierten en URL en este paso.

## 2026-11-11: armado del conjunto y sesgo de forma

Decisión de Andrés. Se conserva todo el phishing y las legítimas se muestrean por estratos de forma (con o sin ruta, esquema y «www.») para imitar al phishing. La proporción es de 0.59 legítimas por cada URL de phishing: usa todas las legítimas con ruta y es la única de las comparadas en la que entra Majestic Million, con el 2.9 % de las legítimas. Las proporciones menores pierden hasta la mitad de las legítimas y casi no reducen la diferencia de forma de las URL con ruta.

Majestic Million solo cubre lo que falta en los estratos sin ruta. Sus dominios salen del top 100,000, sin los que ya aparecen como host, y se escriben con el esquema y el «www.» del estrato.

El sesgo de forma de las URL con ruta no se corrige con el muestreo, porque casi no hay legítimas con ruta en https y sin «www.». Se trata en las características de la Fase 2: el esquema y el «www.» quedan fuera del vector, y el «www.» se quita antes de calcular la longitud y los subdominios. El control de sesgo de fuente lo verifica. Si falla, se agrega Common Crawl como fuente de legítimas con ruta, lo que cambia el Charter y el ADD.
