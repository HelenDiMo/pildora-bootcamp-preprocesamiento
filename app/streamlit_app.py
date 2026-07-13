import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Preprocessing & ML Demo",
    layout="wide"
)

st.title("🐧 Preprocessing & ML Demo — Palmer Penguins")
st.write("Esta demo interactiva muestra técnicas avanzadas de preprocesamiento y comparación entre Random Forest y XGBoost.")

st.sidebar.title("Navegación")
section = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "1. Cargar Dataset",
        "2. Outliers",
        "3. Feature Engineering",
        "4. Preprocesamiento",
        "5. Random Forest",
        "6. XGBoost",
        "7. Comparativa",
    ]
)

if section == "1. Cargar Dataset":
    st.header("1. Cargar Dataset")
    st.code("""
df = sns.load_dataset('penguins')
print("Shape:", df.shape)
df.head()
""")
    df = sns.load_dataset('penguins')
    st.write(df.head())
