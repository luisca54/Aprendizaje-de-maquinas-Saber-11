# U4A1 — CRISP-DM — Predicción de PUNT_GLOBAL Saber 11° 2020-2

Proyecto académico de aprendizaje automático para estimar `PUNT_GLOBAL` de Saber 11° a partir de variables sociodemográficas, familiares y del establecimiento educativo disponibles antes de conocer el resultado.

## Modelo desplegado

El modelo utilizado en Streamlit es un **Voting Regressor** que combina:

- Regresión Lineal
- Ridge
- Random Forest

Desempeño en el conjunto de prueba:

- MAE: **31,11 puntos**
- RMSE: **39,05 puntos**
- R²: **0,357**

El modelo se entrenó con una muestra de modelado de 30.000 registros: 21.000 para entrenamiento y 9.000 para prueba.

## Aplicación Streamlit

`app.py` carga el pipeline serializado `modelo_saber11_voting_final.pkl` y las categorías almacenadas en `metadata.json` para generar predicciones sobre nuevos registros.

La salida es una estimación académica y no corresponde a un resultado oficial del ICFES.

## Archivos del repositorio

- `app.py` — aplicación Streamlit.
- `modelo_saber11_voting_final.pkl` — pipeline final serializado.
- `metadata.json` — variables y categorías utilizadas por la aplicación.
- `requirements.txt` — dependencias necesarias para ejecutar la aplicación.
- `colab_U4A1_CRISP_DM_Saber11_FINAL.ipynb` — notebook del proyecto.
- `README.md` — documentación básica del repositorio.

## Fuente de datos

Datos Abiertos Colombia / ICFES — Saber 11° 2020-2, dataset `rnvb-vnyh`.

https://www.datos.gov.co/en/en/Education/Saber-11-2020-2/rnvb-vnyh

## Nota sobre los entregables

El documento CRISP-DM, el dataset completo y el reporte HTML de `ydata-profiling` se entregan como artefactos académicos separados. Las tablas, resultados y figuras utilizadas para sustentar el modelamiento están documentadas en el informe y en el notebook.
