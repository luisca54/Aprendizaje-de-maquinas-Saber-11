import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Predicción Saber 11",
    page_icon="📚",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_saber11_voting_final.pkl"
META_PATH = BASE_DIR / "metadata.json"

model = joblib.load(MODEL_PATH)
with META_PATH.open(encoding="utf-8") as f:
    meta = json.load(f)

features = meta["features"]
categories = meta["categories"]

labels = {
    "ESTU_GENERO": "Género del estudiante",
    "ESTU_TIENEETNIA": "Tiene pertenencia étnica",
    "ESTU_DEPTO_RESIDE": "Departamento de residencia",
    "FAMI_ESTRATOVIVIENDA": "Estrato de vivienda",
    "FAMI_PERSONASHOGAR": "Personas en el hogar",
    "FAMI_CUARTOSHOGAR": "Cuartos del hogar",
    "FAMI_EDUCACIONPADRE": "Educación del padre",
    "FAMI_EDUCACIONMADRE": "Educación de la madre",
    "FAMI_TRABAJOLABORPADRE": "Trabajo del padre",
    "FAMI_TRABAJOLABORMADRE": "Trabajo de la madre",
    "FAMI_TIENEINTERNET": "Internet en el hogar",
    "FAMI_TIENECOMPUTADOR": "Computador",
    "FAMI_TIENELAVADORA": "Lavadora",
    "FAMI_TIENEAUTOMOVIL": "Automóvil",
    "FAMI_TIENEMOTOCICLETA": "Motocicleta",
    "FAMI_TIENECCONSOLAVIDEOJUEGOS": "Consola de videojuegos",
    "FAMI_TIENECCONSOLAVIDEOJUEGOS": "Consola de videojuegos",
    "FAMI_TIENECONSOLAVIDEOJUEGOS": "Consola de videojuegos",
    "FAMI_NUMLIBROS": "Libros en el hogar",
    "ESTU_DEDICACIONLECTURADIARIA": "Dedicación diaria a lectura",
    "ESTU_DEDICACIONINTERNET": "Dedicación a internet",
    "ESTU_HORASSEMANATRABAJA": "Horas semanales de trabajo",
    "COLE_NATURALEZA": "Naturaleza del establecimiento",
    "COLE_CALENDARIO": "Calendario académico",
    "COLE_BILINGUE": "Bilingüismo",
    "COLE_CARACTER": "Carácter del establecimiento",
    "COLE_AREA_UBICACION": "Zona de ubicación",
    "COLE_JORNADA": "Jornada",
    "COLE_GENERO": "Género del establecimiento",
}

st.title("📚 Predicción del Puntaje Global — Saber 11°")
st.caption(
    "Proyecto académico CRISP-DM. La salida es una estimación estadística y no constituye un resultado oficial del ICFES."
)

with st.form("prediction_form"):
    cols = st.columns(2)
    values = {}

    for i, feature in enumerate(features):
        options = categories[feature]
        values[feature] = cols[i % 2].selectbox(
            labels.get(feature, feature),
            options,
            index=0,
            key=feature,
        )

    submitted = st.form_submit_button("Estimar puntaje")

if submitted:
    X_new = pd.DataFrame([values])
    prediction = float(model.predict(X_new)[0])
    prediction = max(0.0, min(500.0, prediction))

    st.success(f"Puntaje global estimado: **{prediction:.1f} / 500**")
    st.metric("Puntaje estimado", f"{prediction:.1f}")
    st.info(
        "La estimación utiliza características sociodemográficas, familiares y del establecimiento. "
        "No utiliza puntajes de las áreas del examen como entradas."
    )

with st.expander("Sobre el modelo"):
    st.write(
        "Modelo final: Voting Regressor que combina Regresión Lineal, Ridge y Random Forest. "
        "Los modelos clásicos fueron ajustados con RandomizedSearchCV y 5-fold CV; posteriormente "
        "las configuraciones seleccionadas se reentrenaron y el Voting se utilizó para la inferencia."
    )
    st.write("Desempeño final en test: MAE = 31,11 puntos · RMSE = 39,05 · R² = 0,357.")
    st.write(
        "El modelo fue construido con una muestra de modelado de 30.000 registros, "
        "con 21.000 para entrenamiento y 9.000 para prueba."
    )
