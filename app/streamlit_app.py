import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from components.run_button import run_button

from logic.load_data import load_penguins
from logic.outliers import cap_outliers
from logic.feature_engineering import add_bill_ratio
from logic.preprocessing import encoding, split_train_test, build_preprocessor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Inicialización de claves para evitar errores
st.session_state.setdefault("rf_acc", None)
st.session_state.setdefault("xgb_acc", None)



# 1. TITULO

st.set_page_config(page_title="Preprocessing & ML Demo", layout="wide")

st.title("🐧 Preprocessing & ML Demo — Palmer Penguins")
st.write(
    "Esta demo interactiva muestra técnicas avanzadas de preprocesamiento y comparación entre Random Forest y XGBoost."
)

# 2. BARRA LATERAL DE NAVEGACIÓN

st.sidebar.title("PASOS A SEGUIR")
section = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "1. Cargar Dataset y Exploración Inicial",
        "2. Outliers",
        "3. Feature Engineering",
        "4. Preprocesamiento",
        "5. Random Forest",
        "6. XGBoost",
        "7. Comparativa",
        "8. Conclusiones y análisis final",
        "9. Ficha de Criterio Ético"
    ],
)

# Si el usuario no carga primero el dataset le sale el mensaje de error.

if section not in[
    "1. Cargar Dataset y Exploración Inicial",
    "9. Ficha de Criterio Ético"
] and "df" not in st.session_state:
    st.warning("Primero debes cargar el dataset en la sección 1.")
    st.stop()

# SECCION 1: CARGAR DATASET Y EXPLORACIÓN INICIAL

if section == "1. Cargar Dataset y Exploración Inicial":
    st.header("Cargar Dataset")

    df = st.session_state.get("df")

    # ---------------------------------------------------------
    # BLOQUE 1 — Carga del dataset
    # ---------------------------------------------------------
    st.code("""
df = sns.load_dataset('penguins')
print("Shape:", df.shape)
df.head()
""")

    if run_button("Cargar datos"):
        df = load_penguins()
        st.session_state["df"] = df
        st.write("Shape:", df.shape)
        st.write(df.head())
        st.success("Dataset cargado correctamente.")


    # ---------------------------------------------------------
    # BLOQUE 2 — Análisis de nulos
    # ---------------------------------------------------------
    st.header("Análisis de Nulos")
    st.code("""
df.isnull().sum()
""")

    if run_button("Mostrar nulos"):
        df = st.session_state["df"]
        st.write(df.isnull().sum())
        st.success("Análisis de nulos mostrado correctamente. " \
                "Nuestra variable objetivo y - species no contiene nulos, por lo que no necesitamos hacer nada más")

    # ---------------------------------------------------------
    # BLOQUE 4 — Descripción de variables numéricas
    # ---------------------------------------------------------

    st.header("Descripción de variables numéricas")
    st.code("""
df.describe()
""")

    if run_button("Describir variables numéricas"):
        df = st.session_state["df"]
        st.write(df.describe())
        st.success("Descripción generada correctamente.")

# SECCION 2: OUTLIERS
if section == "2. Outliers":
    st.header("2. Outliers")

    st.code("""
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(y=df['bill_length_mm'], ax=ax[0])
ax[0].set_title('Antes de tratar outliers')

Q1 = df['bill_length_mm'].quantile(0.25)
Q3 = df['bill_length_mm'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['bill_length_mm'] < lower_bound) | (df['bill_length_mm'] > upper_bound)]
print(f"Outliers detectados en bill_length_mm: {len(outliers)}")
print(f"Límites válidos: [{lower_bound:.2f}, {upper_bound:.2f}]")

# Tratamiento: "capping" (winsorizing) en vez de eliminar filas
df['bill_length_mm'] = df['bill_length_mm'].clip(lower_bound, upper_bound)

sns.boxplot(y=df['bill_length_mm'], ax=ax[1])
ax[1].set_title('Después de capar outliers')
plt.tight_layout()
plt.show()
""")

    df = st.session_state.get("df")

    if run_button("Tratar outliers"):

        col1, col2 = st.columns(2)

        # Antes del tratamiento → boxplot original
        with col1:
            fig_before, ax = plt.subplots(figsize=(3, 2))
            sns.boxplot(y=df["bill_length_mm"], ax=ax)
            ax.set_title("Antes de tratar outliers")
            st.pyplot(fig_before)

        # Aplicamos el tratamiento
        df, lower, upper = cap_outliers(df, "bill_length_mm")
        st.session_state["df"] = df

        # Después del tratamiento → boxplot corregido
        with col2:
            fig_after, ax2 = plt.subplots(figsize=(3, 2))
            sns.boxplot(y=df["bill_length_mm"], ax=ax2)
            ax2.set_title("Después de capar outliers")
            st.pyplot(fig_after)

        # Información adicional
        st.write(f"Límite inferior: {lower:.2f}")
        st.write(f"Límite superior: {upper:.2f}")

        st.success("Outliers tratados correctamente")

# SECCION 3: FEATURE ENGINEERING

if section == "3. Feature Engineering":
    st.header("3. Feature Engineering")

    st.code("""
df['bill_ratio'] = df['bill_length_mm'] / df['bill_depth_mm']

sns.boxplot(data=df, x='species', y='bill_ratio')
plt.title('bill_ratio por especie')
plt.show()
            """)

    df = st.session_state.get("df")

    if run_button("Crear feature bill_ratio"):
        
        # Aplicamos feature engineering
        df = add_bill_ratio(df)
        st.session_state["df"] = df

        # Gráfica en una fila (igual que en sección 2)
        col1, col2 = st.columns(2)

        with col1:
            st.write("Distribución de bill_ratio por especie")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.boxplot(data=df, x="species", y="bill_ratio", ax=ax)
            ax.set_title("bill_ratio por especie")
            st.pyplot(fig)

        with col2:
            st.write("Relación bill_length_mm vs bill_depth_mm")
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            sns.scatterplot(data=df, x="bill_length_mm", y="bill_depth_mm", hue="species", ax=ax2)
            ax2.set_title("Relación original del pico")
            st.pyplot(fig2)

        st.success("Feature 'bill_ratio' creada correctamente")

# SECCION 4: PREPROCESAMIENTO

if section == "4. Preprocesamiento":

# ---------------------------------------------------------
# BLOQUE 1 — Encoding
# ---------------------------------------------------------

    st.header("Encoding")

    st.code("""
num_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'bill_ratio']
cat_cols = ['island', 'sex']

X = df[num_cols + cat_cols]
y = df['species']
""")

    df = st.session_state.get("df")

    if run_button("Realizar Encoding"):
        # Lógica real en preprocessor.py
        X, y, num_cols, cat_cols = encoding(df)

        # Guardamos en session_state
        st.session_state["X"] = X
        st.session_state["y"] = y
        st.session_state["num_cols"] = num_cols
        st.session_state["cat_cols"] = cat_cols

        st.write("X shape:", X.shape)
        st.write("y shape:", y.shape)
        st.success("Encoding realizado correctamente.")

# ---------------------------------------------------------
# BLOQUE 2 — Split Train/Test
# ---------------------------------------------------------
    st.header("Split Train/Test")

    st.code("""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
)

    print("Train:", X_train.shape)
    print("Test:", X_test.shape)
    """)

    #df = st.session_state.get("df")

    if run_button("Realizar Split"):
        # Recuperamos X e y generados en el bloque anterior
        X = st.session_state["X"]
        y = st.session_state["y"]

        # Lógica real en preprocessor.py
        X_train, X_test, y_train, y_test = split_train_test(X, y)

        # Guardamos en session_state
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test

        # Mostramos shapes
        st.write("Train:", X_train.shape)
        st.write("Test:", X_test.shape)

        st.success("Split realizado correctamente.")

# ---------------------------------------------------------
# BLOQUE 3 — Preprocessor
# ---------------------------------------------------------
    st.header("Preprocessor")

    st.code("""
    preprocessor = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ])
    """)

    #df = st.session_state.get("df")

    if run_button("Crear Preprocessor"):
        # Recuperamos columnas del bloque anterior
        num_cols = st.session_state["num_cols"]
        cat_cols = st.session_state["cat_cols"]

        # Lógica real en preprocessor.py
        preprocessor = build_preprocessor(num_cols, cat_cols)

        # Guardamos en session_state
        st.session_state["preprocessor"] = preprocessor

        st.success("Preprocessor creado correctamente.")

# SECCION 5: RANDOM FOREST

if section == "5. Random Forest":
    st.header("5. Random Forest")

    st.code("""
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=200, random_state=42))
])

rf_pipeline.fit(X_train, y_train)

rf_preds = rf_pipeline.predict(X_test)

print("Random Forest - Accuracy:", accuracy_score(y_test, rf_preds))
print(classification_report(y_test, rf_preds))
""")

    if run_button("Entrenar Random Forest"):

        # Recuperamos datos y preprocessor
        ss = st.session_state
        X_train, X_test = ss["X_train"], ss["X_test"]
        y_train, y_test = ss["y_train"], ss["y_test"]
        preprocessor = ss["preprocessor"]


        # Construimos el pipeline real
        rf_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestClassifier(n_estimators=200, random_state=42))
        ])

        # Entrenamos
        rf_pipeline.fit(X_train, y_train)
        rf_preds = rf_pipeline.predict(X_test)

        # Métricas
        acc = accuracy_score(y_test, rf_preds)
        st.session_state["rf_acc"] = acc
        report = classification_report(y_test, rf_preds, output_dict=True)

        st.write(f"### Accuracy: **{acc:.4f}**")

        # Mostramos clasificación por especie
        st.write("### Clasificación por especie")
        st.write(pd.DataFrame(report).transpose())

        # Matriz de confusión
        st.write("### Matriz de confusión")
        fig, ax = plt.subplots(figsize=(3, 2))
        ConfusionMatrixDisplay.from_predictions(y_test, rf_preds, ax=ax)
        st.pyplot(fig)

        st.success("Random Forest entrenado correctamente.")

# SECCION 6: XGBOOST
if section == "6. XGBoost":
    st.header("6. XGBoost")

    st.code("""
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    ))
])

xgb_pipeline.fit(X_train, y_train_enc)

xgb_preds = xgb_pipeline.predict(X_test)

print("XGBoost - Accuracy:", accuracy_score(y_test_enc, xgb_preds))
print(classification_report(y_test_enc, xgb_preds))
""")

    if run_button("Entrenar XGBoost"):

        # Recuperamos datos y preprocessor
        ss = st.session_state
        X_train, X_test = ss["X_train"], ss["X_test"]
        y_train, y_test = ss["y_train"], ss["y_test"]
        preprocessor = ss["preprocessor"]

        # Codificamos y
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        # Construimos el pipeline real
        xgb_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                eval_metric='mlogloss'
            ))
        ])

        # Entrenamos
        xgb_pipeline.fit(X_train, y_train_enc)
        xgb_preds = xgb_pipeline.predict(X_test)

        # Métricas
        acc = accuracy_score(y_test_enc, xgb_preds)
        st.session_state["xgb_acc"] = acc


        report = classification_report(y_test_enc, xgb_preds, output_dict=True)

        st.write(f"### Accuracy: **{acc:.4f}**")

        # Mostramos clasificación por especie
        st.write("### Clasificación por especie")
        st.write(pd.DataFrame(report).transpose())

        # Matriz de confusión
        st.write("### Matriz de confusión")
        fig, ax = plt.subplots(figsize=(4, 3))
        ConfusionMatrixDisplay.from_predictions(y_test_enc, xgb_preds, ax=ax)
        st.pyplot(fig)

        st.success("XGBoost entrenado correctamente.")

# ---------------------------------------------------------
# BLOQUE 7 — Comparativa de modelos
# ---------------------------------------------------------

if section == "7. Comparativa":
    # ---------------------------------------------------------
# BLOQUE 7 — Comparativa de modelos
# ---------------------------------------------------------
    st.header("Comparativa Random Forest vs XGBoost")

    st.code("""
rf_acc = accuracy_score(y_test, rf_preds)
xgb_acc = accuracy_score(y_test_enc, xgb_preds)

plt.bar(['Random Forest', 'XGBoost'], [rf_acc, xgb_acc])
plt.title('Comparativa de Accuracy')
plt.show()
""")

    if run_button("Comparar modelos"):

        rf_acc = st.session_state.get("rf_acc", None)
        xgb_acc = st.session_state.get("xgb_acc", None)

    # Validación robusta
        if rf_acc is None:
            st.error("❌ Debes entrenar el modelo Random Forest antes de comparar.")
            st.stop()

        if xgb_acc is None:
            st.error("❌ Debes entrenar el modelo XGBoost antes de comparar.")
            st.stop()

    # Tabla comparativa
        st.write("### Tabla comparativa")
        df_comp = pd.DataFrame({
            "Modelo": ["Random Forest", "XGBoost"],
            "Accuracy": [rf_acc, xgb_acc]
        })
        st.write(df_comp)

    # Gráfica comparativa
        st.write("### Gráfica de Accuracy")
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(["Random Forest", "XGBoost"], [rf_acc, xgb_acc], color=["steelblue", "orange"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Comparativa de modelos")
        st.pyplot(fig)

    # Recomendación automática
        st.write("### Mejor modelo")
        if xgb_acc > rf_acc:
            st.success(f"XGBoost es el mejor modelo con accuracy {xgb_acc:.4f}")
        elif rf_acc > xgb_acc:
            st.success(f"Random Forest es el mejor modelo con accuracy {rf_acc:.4f}")
        else:
            st.info("Ambos modelos tienen el mismo rendimiento.")

# ---------------------------------------------------------
# SECCIÓN 8 — Conclusiones y análisis final
# ---------------------------------------------------------
if section == "8. Conclusiones y análisis final":
    st.header("Conclusiones del análisis")

    st.code("""
    print("Random Forest Accuracy:", rf_acc)
    print("XGBoost Accuracy:", xgb_acc)

    if xgb_acc > rf_acc:
        print("XGBoost es el mejor modelo.")
    elif rf_acc > xgb_acc:
        print("Random Forest es el mejor modelo.")
    else:
        print("Ambos modelos tienen el mismo rendimiento.")
    """)

    if run_button("Mostrar conclusiones"):

        rf_acc = st.session_state.get("rf_acc", None)
        xgb_acc = st.session_state.get("xgb_acc", None)

    # Validación robusta
        if rf_acc is None or xgb_acc is None:
            st.error("❌ Debes entrenar ambos modelos antes de ver las conclusiones.")
            st.stop()

    # Resumen numérico
        st.write("### Resumen de métricas")
        st.write(f"- **Random Forest Accuracy:** {rf_acc:.4f}")
        st.write(f"- **XGBoost Accuracy:** {xgb_acc:.4f}")

    # Interpretación
        st.write("### Interpretación de resultados")

        if xgb_acc > rf_acc:
            st.success("**XGBoost es el modelo con mejor rendimiento en este dataset.**")
            st.write("""
    XGBoost suele destacar cuando:
    - Hay relaciones no lineales complejas.
    - El dataset tiene ruido.
    - Las variables tienen interacciones fuertes.
    - Se necesita un modelo más robusto y con mejor generalización.

    En este caso, XGBoost ha capturado mejor las diferencias entre especies.
    """)
        elif rf_acc > xgb_acc:
            st.success("**Random Forest es el modelo con mejor rendimiento en este dataset.**")
            st.write("""
Random Forest suele destacar cuando:
- El dataset es pequeño o mediano.
- Las clases están bien separadas.
- El preprocesamiento es sencillo.
- Se busca un modelo estable y fácil de interpretar.

Aquí, Random Forest ha sido suficiente para capturar las diferencias entre especies.
""")
        else:
            st.info("**Ambos modelos tienen el mismo rendimiento.**")
            st.write("""
Esto indica que:
- El dataset es sencillo de separar.
- Ambos modelos capturan bien las relaciones.
- No hay una ventaja clara en complejidad o capacidad de generalización.
""")

    # Recomendaciones finales
        st.write("### Recomendaciones para producción")

        st.write("""
- **Random Forest**: más simple, más estable, menos propenso a sobreajuste.
- **XGBoost**: más potente, mejor rendimiento en datasets complejos, más configurable.
- Para producción:
  - Si buscas **simplicidad y estabilidad**, usa Random Forest.
  - Si buscas **máxima precisión**, usa XGBoost.
  - Si el dataset crece, XGBoost suele escalar mejor.
""")

        st.success("Conclusiones generadas correctamente.")

# ---------------------------------------------------------
# SECCIÓN 9 — Ficha de Criterio Ético
# ---------------------------------------------------------

if section == "9. Ficha de Criterio Ético":
    
    st.markdown("""
## 🧭 Criterio Ético y Limitaciones del Modelo

### 📌 Contexto del dataset
El dataset **Palmer Penguins** fue recopilado por la Dra. Kristen Gorman en la Estación Palmer (Antártida).  
Incluye **344 observaciones** de **tres especies** de pingüinos en **tres islas** concretas.

Aunque es excelente para fines pedagógicos, presenta limitaciones claras:

- Procede de **una única región geográfica**.  
- Contiene **pocas especies** y **pocas muestras**.  
- Los modelos entrenados con él **no son generalizables** a otras poblaciones o ecosistemas.

---

### 🤖 Riesgos algorítmicos
Modelos como **Random Forest** y **XGBoost** pueden:

- Sobreajustar fácilmente en datasets pequeños (de hecho, aquí se alcanza **100% accuracy**).  
- Actuar como **cajas negras**, dificultando explicar por qué predicen lo que predicen.  
- Requerir validación adicional (p. ej., **cross-validation**) antes de confiar en sus resultados.

---

### ⚠️ Riesgos si este flujo se aplicara a datos sensibles
Aunque este dataset es inocuo, **el mismo pipeline** aplicado a datos reales podría causar:

- **Sesgos amplificados** por mala limpieza de outliers.  
- **Fugas de información** entre train/test que inflan métricas.  
- **Decisiones injustas** si se confía solo en accuracy.  
- Impacto negativo en ámbitos como:
  - salud,  
  - concesión de crédito,  
  - selección de personal,  
  - evaluación de riesgo.

---

### 📝 Nota final
Este bloque es una **reflexión ética complementaria** para acompañar la entrega técnica.  
No sustituye un análisis formal de impacto algorítmico.
""")
