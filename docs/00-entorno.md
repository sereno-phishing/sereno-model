# Entorno

Equipo oficial de entrenamiento: Intel Core i7-12650H, 16 hilos, 31.6 GB de RAM reportados por Windows, Windows 11 y Python 3.12.10. Corre en CPU.

## Instalar

Desde la raíz del repositorio, en Git Bash:

```bash
python -m venv "$USERPROFILE/.venvs/sereno-model"
"$USERPROFILE/.venvs/sereno-model/Scripts/python.exe" -m pip install -r requirements.txt
"$USERPROFILE/.venvs/sereno-model/Scripts/python.exe" -m pip install -e . --no-deps
```

## Verificar

```bash
"$USERPROFILE/.venvs/sereno-model/Scripts/python.exe" -m pytest -q
```

`tests/test_environment.py` entrena un Stacking con los cuatro algoritmos del ADD sobre datos sintéticos. Comprueba que dos corridas con la misma semilla den las mismas probabilidades y que el modelo guardado prediga lo mismo. Con esto se valida el entorno, no el modelo real.

Resultado del 2026-11-03: 2 pruebas pasan.

## Versiones

`requirements.in` lista las dependencias directas y `requirements.txt` fija las versiones exactas. Las principales:

| Librería | Versión |
|---|---|
| scikit-learn | 1.9.1 |
| xgboost | 3.4.1 |
| lightgbm | 4.7.0 |
| numpy | 2.5.4 |
| pandas | 3.1.0 |
| joblib | 1.6.0 |

`sereno-backend` debe usar las mismas versiones de las librerías que cargan el modelo. Si cambia una versión, se vuelve a correr la verificación y se registra el motivo en `decisiones.md`.
