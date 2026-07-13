import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from components.code_viewer import show_code
from components.run_button import run_button

from logic.load_data import load_penguins
from logic.outliers import cap_outliers
from logic.feature_engineering import add_bill_ratio
from logic.preprocessing import build_preprocessor
from logic.models import train_random_forest, train_xgboost

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
