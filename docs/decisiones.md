# Decisiones del modelo

Formato: fecha, decisión, motivo y alternativas descartadas.

## 2026-11-03: repositorio propio

`sereno-model` vive aparte de `sereno-backend`, al mismo nivel. El ADD separa la estación de entrenamiento del servidor de inferencia. Se descartó una carpeta dentro del backend porque mezclaría datasets, experimentos y dependencias de análisis con el servicio.

## 2026-11-03: equipo de entrenamiento

El entrenamiento oficial se ejecuta solo en el equipo de Andrés: Intel Core i7-12650H, 16 hilos, 31.6 GB de RAM, Windows y Python 3.12. Corre en CPU, como indica el ADD.

La GPU (RTX 3070 Laptop, 8 GB) no se usa en el entrenamiento oficial. Random Forest no la aprovecha y LightGBM en Windows solo admite la versión OpenCL, que acelera parte del proceso y por defecto usa precisión de 32 bits. Queda como apoyo opcional para explorar hiperparámetros con XGBoost, si se mide que hace falta.

## 2026-11-03: ubicación del entorno y los datos

El entorno virtual vive fuera de OneDrive, en `%USERPROFILE%\.venvs\sereno-model`, para no sincronizar miles de archivos. Las descargas originales quedan en `data/raw/`, dentro de OneDrive, porque los feeds de phishing no se pueden volver a descargar iguales y así tienen respaldo.
