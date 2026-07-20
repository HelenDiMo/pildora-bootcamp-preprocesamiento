import os
import sys
import streamlit as st
import matplotlib.pyplot as plt  # <--- Esta es la línea que te falta
import seaborn as sns

# Conseguimos la ruta absoluta del directorio donde está ESTE archivo (la carpeta 'app')
base_path = os.path.dirname(os.path.abspath(__file__))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

# Ahora las importaciones funcionarán de forma idéntica en local y en la nube
from components.buttons import run_button, retro_typewriter_code
from components.code_viewer import colorear_codigo

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

st.set_page_config(page_title=" 🐧 Preprocessing & ML Demo", layout="wide")

# ======== CARGAR CSS ========
with open("app/assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    

# ======== BANNER ========
st.image("app/assets/banner.jpg", width="stretch")

st.title("PREPROCESSING & ML DEMO — Palmer Penguins")
st.write("""
    *Demo interactiva para mostrar técnicas avanzadas de preprocesamiento y comparación entre Random Forest y XGBoost.*
    - **Dataset Utilizado**: Palmer penguins
""")

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

# =========================================================================
# SECCION 1: CARGAR DATASET Y EXPLORACIÓN INICIAL
# =========================================================================

if section == "1. Cargar Dataset y Exploración Inicial":
    st.header("Cargar Dataset")

    # 1. Creamos un estado persistente para saber si el resultado ya se ejecutó
    if "df_cargado" not in st.session_state:
        st.session_state["df_cargado"] = False

    df = st.session_state.get("df")

    # ---------------------------------------------------------
    # BLOQUE 1 — Carga del dataset (Efecto máquina de Escribir)
    # ---------------------------------------------------------
    codigo_puro_1_1 = ("""df = sns.load_dataset('penguins')
print("Shape:", df.shape)
df.head()"""
    )

    codigo_coloreado_1_1 = colorear_codigo(codigo_puro_1_1)
    
    # Invocamos la función mágica (se queda exactamente igual)
    retro_typewriter_code(codigo_coloreado_1_1, key="key_carga", height=180) 
    
    # 2. Si pulsas el botón, guardamos el DataFrame y activamos el interruptor
    if run_button("Cargar Dataset"):
        df = load_penguins()
        st.session_state["df"] = df
        st.session_state["df_cargado"] = True

    # 3. PERSISTENCIA: Si el interruptor está activo, el resultado se queda fijo en pantalla
    if st.session_state["df_cargado"]:
        df = st.session_state["df"]  # Recuperamos los datos guardados
        st.write("Shape:", df.shape)
        st.write(df.head())
        st.success("Dataset cargado correctamente 🐧")

    # ---------------------------------------------------------
    # BLOQUE 2 — Análisis de nulos
    # ---------------------------------------------------------
    st.header("Análisis de Nulos")

    # Inicializamos el estado persistente para el resultado del Bloque 2
    if "nulos_mostrados" not in st.session_state:
        st.session_state["nulos_mostrados"] = False

    codigo_seccion_1_2 = (
        "df.isnull().sum()"
        )
    
    codigo_coloreado_1_2 = colorear_codigo(codigo_seccion_1_2)

    retro_typewriter_code(codigo_coloreado_1_2, key="key_nulos_col", height=100) # en este height hago que queda el codigo en el contenedor

    # Si se pulsa el botón, activamos su persistencia
    if run_button("Mostrar nulos"):
        st.session_state["nulos_mostrados"] = True

    # Renderizado persistente del resultado del Bloque 2
    if st.session_state["nulos_mostrados"]:
        df = st.session_state["df"]
        st.write(df.isnull().sum())
        st.success("Análisis de nulos mostrado correctamente."
                "\nNuestra variable objetivo y - species no contiene nulos, por lo que no necesitamos hacer nada más")

    # ---------------------------------------------------------
    # BLOQUE 3 — Descripción de variables numéricas
    # ---------------------------------------------------------

    st.header("Descripción de variables numéricas")
    codigo_seccion_1_3 = (
        "df.describe()"
        )
        
    codigo_coloreado_1_3 = colorear_codigo(codigo_seccion_1_3)

    
    retro_typewriter_code(codigo_coloreado_1_3, key="key_descripcion", height=100) # en este height hagoq ue queda el codigo en el contenedor

    if run_button("Describir variables numéricas"):
        df = st.session_state["df"]
        st.write(df.describe())
        st.success("Descripción generada correctamente.")

# =========================================================================================
# SECCION 2: OUTLIERS
# =========================================================================

if section == "2. Outliers":
    st.header("2. Outliers")

    # 1. Inicializamos el estado persistente para los gráficos y resultados de esta sección
    if "outliers_tratados" not in st.session_state:
        st.session_state["outliers_tratados"] = False

    codigo_seccion_2 = """fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(y=df['bill_length_mm'], ax=ax[0])
ax[0].set_title('Antes de tratar outliers')

Q1 = df['bill_length_mm'].quantile(0.25)
Q3 = df['bill_length_mm'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['bill_length_mm'] < lower_bound) | 
    (df['bill_length_mm'] > upper_bound)]

print(f"Outliers detectados en bill_length_mm: {len(outliers)}")
print(f"Límites válidos: [{lower_bound:.2f}, {upper_bound:.2f}]")

# Tratamiento: "capping" (winsorizing) en vez de eliminar filas
df['bill_length_mm'] = df['bill_length_mm'].clip(lower_bound, upper_bound)

sns.boxplot(y=df['bill_length_mm'], ax=ax[1])
ax[1].set_title('Después de capar outliers')
plt.tight_layout()
plt.show()"""

    codigo_coloreado_2 = colorear_codigo(codigo_seccion_2)

    retro_typewriter_code(codigo_coloreado_2, key="key_outliers", height=900) # en este height hagoq ue queda el codigo en el contenedor

    # 3. Al pulsar el botón, ejecutamos el tratamiento matemático y guardamos los resultados en el estado
    if run_button("Tratar outliers"):
        df_actual = st.session_state.get("df")
        
        # Aplicamos tu función de tratamiento
        df_procesado, lower, upper = cap_outliers(df_actual, "bill_length_mm")
        
        # Guardamos en sesión tanto el nuevo DataFrame como los límites calculados
        st.session_state["df"] = df_procesado
        st.session_state["outliers_lower"] = lower
        st.session_state["outliers_upper"] = upper
        
        # Activamos el interruptor de persistencia
        st.session_state["outliers_tratados"] = True

# 4. PERSISTENCIA: Si el interruptor está activo, dibujamos los gráficos de forma fija
    if st.session_state["outliers_tratados"]:
        # 1. Recuperamos el DataFrame del estado de la sesión de forma segura
        # Si 'df' mutó en el botón, podemos usar una copia o el df_actual que recuperaste arriba
        df_persistente = st.session_state["df"]
        lower = st.session_state["outliers_lower"]
        upper = st.session_state["outliers_upper"]

        # 2. Para contar cuántos outliers HABÍA, usamos el mismo DataFrame persistente.
        # Al usar .clip(), los valores extremos se igualan a los límites, por lo que podemos
        # contar cuántos valores coinciden exactamente con los límites (el "capping" aplicado)
        outliers_inferiores = df_persistente['bill_length_mm'] == lower
        outliers_superiores = df_persistente['bill_length_mm'] == upper
        
        cant_outliers = outliers_inferiores.sum() + outliers_superiores.sum()

        col1, col2 = st.columns(2)

        # Gráfico 1: Para mostrar el "Antes" de forma persistente, dado que df_original no existe globalmente,
        # lo ideal es que en la SECCION 1 (o al cargar el archivo) guardes un "df_original" en st.session_state.
        # Si no lo tienes guardado, de momento renderizamos el estado actual:
        with col1:
            fig_before, ax = plt.subplots(figsize=(3, 2))
            # Si guardas el original al inicio del script como st.session_state["df_original"], úsalo aquí
            if "df_original" in st.session_state:
                sns.boxplot(y=st.session_state["df_original"]["bill_length_mm"], ax=ax)
            else:
                sns.boxplot(y=df_persistente["bill_length_mm"], ax=ax)
            ax.set_title("Antes de tratar outliers")
            st.pyplot(fig_before)

        # Gráfico 2: Renderizado del boxplot posterior (ya capado)
        with col2:
            fig_after, ax2 = plt.subplots(figsize=(3, 2))
            sns.boxplot(y=df_persistente["bill_length_mm"], ax=ax2)
            ax2.set_title("Después de capar outliers")
            st.pyplot(fig_after)

        # 4. Información adicional y mensaje de éxito fijos (Usando la variable calculada arriba)
        st.write(f"Outliers detectados en bill_length_mm: **{cant_outliers}**")
        st.write(f"Límite inferior: {lower:.2f}")
        st.write(f"Límite superior: {upper:.2f}")
        st.success("Outliers tratados correctamente")

# =========================================================================================
# SECCION 3: FEATURE ENGINEERING
# =========================================================================================


if section == "3. Feature Engineering":
    st.header("3. Feature Engineering")

    # 1. Inicializamos el estado persistente para los gráficos y resultados de esta sección
    if "feature_engineering_ejecutado" not in st.session_state:
        st.session_state["feature_engineering_ejecutado"] = False

    codigo_seccion_3 = ("""df['bill_ratio'] = df['bill_length_mm'] / df['bill_depth_mm']

sns.boxplot(data=df, x='species', y='bill_ratio')
plt.title('bill_ratio por especie')
plt.show()""")
    
    codigo_coloreado_3 = colorear_codigo(codigo_seccion_3)

    # Corrección: Pasamos codigo_seccion_3, clave única "key_feature" y ajustamos la altura a 120
    retro_typewriter_code(codigo_coloreado_3, key="key_feature", height=280)

    # 2. Al pulsar el botón, calculamos la nueva feature y activamos la persistencia
    if run_button("Crear feature bill_ratio"):
        df_actual = st.session_state.get("df")
        
        # Aplicamos feature engineering
        df_procesado = add_bill_ratio(df_actual)
        st.session_state["df"] = df_procesado
        
        # Encendemos el interruptor de persistencia
        st.session_state["feature_engineering_ejecutado"] = True

    # 3. PERSISTENCIA: Si ya se ejecutó, mantenemos las dos columnas de gráficos fijas en pantalla
    if st.session_state["feature_engineering_ejecutado"]:
        df_persistente = st.session_state["df"]

        col1, col2 = st.columns(2)

        with col1:
            st.write("Distribución de bill_ratio por especie")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.boxplot(data=df_persistente, x="species", y="bill_ratio", ax=ax)
            ax.set_title("bill_ratio por especie")
            st.pyplot(fig)

        with col2:
            st.write("Relación bill_length_mm vs bill_depth_mm")
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            sns.scatterplot(data=df_persistente, x="bill_length_mm", y="bill_depth_mm", hue="species", ax=ax2)
            ax2.set_title("Relación original del pico")
            st.pyplot(fig2)

        st.success("Feature 'bill_ratio' creada correctamente")

# SECCION 4: PREPROCESAMIENTO

if section == "4. Preprocesamiento":

# ---------------------------------------------------------
# BLOQUE 1 — Encoding
# ---------------------------------------------------------

    st.header("Encoding")

    # 1. Inicializamos el estado persistente para esta subsección
    if "encoding_ejecutado" not in st.session_state:
        st.session_state["encoding_ejecutado"] = False

    codigo_seccion_4_1 = """num_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 
                                'body_mass_g', 'bill_ratio']
cat_cols = ['island', 'sex']

X = df[num_cols + cat_cols]
y = df['species']"""

    codigo_coloreado_4_1 = colorear_codigo(codigo_seccion_4_1)

    # Pasamos la variable correcta codigo_seccion_4_1 y su clave única
    retro_typewriter_code(codigo_coloreado_4_1, key="key_encoding", height=300)

    # 2. Al pulsar el botón, ejecutamos la transformación y activamos la persistencia
    if run_button("Realizar Encoding"):
        df_actual = st.session_state.get("df")
    
        # Lógica real (llamada a tu función)
        X, y, num_cols, cat_cols = encoding(df_actual)

        # Guardamos absolutamente todo en el session_state
        st.session_state["X"] = X
        st.session_state["y"] = y
        st.session_state["num_cols"] = num_cols
        st.session_state["cat_cols"] = cat_cols
    
        # ¡Encendemos el interruptor!
        st.session_state["encoding_ejecutado"] = True

    # 3. PERSISTENCIA: Mantenemos las dimensiones y el mensaje de éxito fijos en pantalla
    if st.session_state["encoding_ejecutado"]:
        X_persistente = st.session_state["X"]
        y_persistente = st.session_state["y"]

        st.write("X shape:", X_persistente.shape)
        st.write("y shape:", y_persistente.shape)
        st.success("Encoding realizado correctamente.")

# ---------------------------------------------------------
# BLOQUE 2 — Split Train/Test
# ---------------------------------------------------------
    st.header("Split Train/Test")

    # 1. INICIALIZACIÓN: Evita que Streamlit falle si la clave no existe al cargar la sección
    if "split_ejecutado" not in st.session_state:
        st.session_state["split_ejecutado"] = False

    codigo_seccion_4_2 = ("""X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
print("Train:", X_train.shape)
print("Test:", X_test.shape)""")

    codigo_coloreado_4_2 = colorear_codigo(codigo_seccion_4_2)

    # Pasamos la variable correcta codigo_seccion_4_1 y su clave única
    retro_typewriter_code(codigo_coloreado_4_2, key="key_split_train_test", height=230)

    if run_button("Realizar Split"):
        # Comprobamos si X e y existen en el estado antes de continuar
        if "X" not in st.session_state or "y" not in st.session_state:
            st.error("⚠️ No se encuentran las matrices X e y. Por favor, ejecuta primero el bloque anterior de 'Encoding'.")
        else:
            X = st.session_state["X"]
            y = st.session_state["y"]

        # Lógica real en preprocessor.py
        X_train, X_test, y_train, y_test = split_train_test(X, y)

        # Guardamos en session_state
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test

        # ¡Activamos el interruptor!
        st.session_state["split_ejecutado"] = True
    
    # 3. PERSISTENCIA: Renderizado fijo de las dimensiones resultantes
    if st.session_state["split_ejecutado"]:
        X_train_p = st.session_state["X_train"]
        X_test_p = st.session_state["X_test"]

        st.write("Train:", X_train_p.shape)
        st.write("Test:", X_test_p.shape)
        st.success("Split realizado correctamente.")
# ---------------------------------------------------------
# BLOQUE 3 — Preprocessor
# ---------------------------------------------------------
    st.header("Preprocessor")

    # 1. Inicializamos el estado persistente para esta subsección
    if "preprocessor_ejecutado" not in st.session_state:
        st.session_state["preprocessor_ejecutado"] = False

    codigo_seccion_4_3 = ("""preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_cols),  
    ('cat', Pipeline([ 
        ('imputer', SimpleImputer(strategy='most_frequent')),  
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_cols)
])""")

    codigo_coloreado_4_3 = colorear_codigo(codigo_seccion_4_3)

    # Pasamos la variable correcta codigo_seccion_4_1 y su clave única
    retro_typewriter_code(codigo_coloreado_4_3, key="key_preprocessor", height=350)

    # 2. Al pulsar el botón, validamos dependencias, ejecutamos la función y activamos persistencia
    if run_button("Crear Preprocessor"):
        # Comprobamos si las listas de columnas existen en la sesión
        if "num_cols" not in st.session_state or "cat_cols" not in st.session_state:
            st.error("⚠️ No se encuentran las listas de columnas. Por favor, ejecuta primero el bloque de 'Encoding'.")
        else:
            # Recuperamos columnas guardadas en la sesión
            num_cols = st.session_state["num_cols"]
            cat_cols = st.session_state["cat_cols"]

            # Lógica real en tu script de procesamiento
            preprocessor = build_preprocessor(num_cols, cat_cols)

            # Guardamos en session_state de forma persistente
            st.session_state["preprocessor"] = preprocessor
        
            # ¡Encendemos el interruptor!
            st.session_state["preprocessor_ejecutado"] = True

    # 3. PERSISTENCIA: Renderizado del mensaje de éxito para que no desaparezca
    if st.session_state["preprocessor_ejecutado"]:
        st.success("Preprocessor creado correctamente.")


#============================================
# SECCION 5: RANDOM FOREST
#============================================

if section == "5. Random Forest":
    st.header("5. Random Forest")

    # 1. Inicializamos el estado persistente para el modelo y sus métricas
    if "rf_ejecutado" not in st.session_state:
        st.session_state["rf_ejecutado"] = False

    codigo_seccion_5 = """rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=200, random_state=42))
])

rf_pipeline.fit(X_train, y_train)
rf_preds = rf_pipeline.predict(X_test)

print("Random Forest - Accuracy:", accuracy_score(y_test, rf_preds))
print(classification_report(y_test, rf_preds))"""

    codigo_coloreado_5 = colorear_codigo(codigo_seccion_5)

    # Cambiamos st.code por tu componente retro con clave única y ajuste de altura
    retro_typewriter_code(codigo_coloreado_5, key="key_random_forest", height=430)

    # 2. Al pulsar el botón, validamos que existan los datos del split y el preprocesador
    if run_button("Entrenar Random Forest"):
        ss = st.session_state
        
        requisitos = ["X_train", "X_test", "y_train", "y_test", "preprocessor"]
        if not all(k in ss for k in requisitos):
            st.error("⚠️ Faltan dependencias. Asegúrate de haber completado el Split Train/Test y la creación del Preprocessor en la sección 4.")
        else:
            # Recuperamos datos y preprocessor
            X_train, X_test = ss["X_train"], ss["X_test"]
            y_train, y_test = ss["y_train"], ss["y_test"]
            preprocessor = ss["preprocessor"]

            # Construimos el pipeline real y entrenamos
            rf_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', RandomForestClassifier(n_estimators=200, random_state=42))
            ])
            
            rf_pipeline.fit(X_train, y_train)
            rf_preds = rf_pipeline.predict(X_test)

            # Calculamos métricas
            acc = accuracy_score(y_test, rf_preds)
            report = classification_report(y_test, rf_preds, output_dict=True)

            # Guardamos los resultados y predicciones en session_state para la persistencia
            st.session_state["rf_pipeline"] = rf_pipeline
            st.session_state["rf_acc"] = acc
            st.session_state["rf_report"] = pd.DataFrame(report).transpose()
            st.session_state["rf_y_test"] = y_test
            st.session_state["rf_preds"] = rf_preds
            
            # ¡Encendemos el interruptor del modelo!
            st.session_state["rf_ejecutado"] = True

    # 3. PERSISTENCIA: Renderizado fijo de los resultados del entrenamiento 
    if st.session_state.get("rf_ejecutado"):
        st.write(f"### Accuracy: **{st.session_state['rf_acc']:.4f}**")

        # Mostramos clasificación por especie (fijo en pantalla)
        st.write("### Clasificación por especie")
        st.dataframe(st.session_state["rf_report"])

        # Matriz de confusión (se regenera de forma segura a partir de los datos guardados)
        st.write("### Matriz de confusión")
        fig, ax = plt.subplots(figsize=(4, 3))
        ConfusionMatrixDisplay.from_predictions(
            st.session_state["rf_y_test"], 
            st.session_state["rf_preds"], 
            ax=ax,
            cmap="viridis" # Un toque de color genial para el look cyber/retro
        )
        st.pyplot(fig)

        st.success("Random Forest entrenado correctamente.")

#=============================================================
# SECCION 6: XGBOOST
#=============================================================

if section == "6. XGBoost":
    st.header("6. XGBoost")

    # 1. Inicializamos el estado persistente para XGBoost
    if "xgb_ejecutado" not in st.session_state:
        st.session_state["xgb_ejecutado"] = False

    codigo_seccion_6 = """le = LabelEncoder()
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
print(classification_report(y_test_enc, xgb_preds, target_names=le.classes_))"""

    codigo_coloreado_6 = colorear_codigo(codigo_seccion_6)

    # Pintamos PRIMERO la máquina de escribir
    retro_typewriter_code(codigo_coloreado_6, key="key_xgboost", height=800)

    # Pintamos SEGUNDO el botón de ejecución
    if run_button("Entrenar XGBoost"):
        ss = st.session_state
        
        requisitos = ["X_train", "X_test", "y_train", "y_test", "preprocessor"]
        if not all(k in ss for k in requisitos):
            st.error("⚠️ Faltan dependencias. Asegúrate de haber completado el Split Train/Test y la creación del Preprocessor en la sección 4.")
        else:
            X_train, X_test = ss["X_train"], ss["X_test"]
            y_train, y_test = ss["y_train"], ss["y_test"]
            preprocessor = ss["preprocessor"]

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

            preds_decoded = le.inverse_transform(xgb_preds)
            report = classification_report(y_test, preds_decoded, output_dict=True)
            acc = accuracy_score(y_test_enc, xgb_preds)

            # Almacenamos todo en la sesión
            st.session_state["xgb_acc"] = acc
            st.session_state["xgb_report"] = pd.DataFrame(report).transpose()
            st.session_state["xgb_y_test_orig"] = y_test
            st.session_state["xgb_preds_decoded"] = preds_decoded
            
            # Encendemos el interruptor
            st.session_state["xgb_ejecutado"] = True

    # 3. PERSISTENCIA (SIEMPRE AL FINAL): Se renderiza abajo solo si el interruptor está activo
    if st.session_state["xgb_ejecutado"]:
        st.write(f"### Accuracy: **{st.session_state['xgb_acc']:.4f}**")

        st.write("### Clasificación por especie")
        st.dataframe(st.session_state["xgb_report"])

        st.write("### Matriz de confusión")
        fig, ax = plt.subplots(figsize=(4, 3))
        ConfusionMatrixDisplay.from_predictions(
            st.session_state["xgb_y_test_orig"], 
            st.session_state["xgb_preds_decoded"], 
            ax=ax,
            cmap="plasma"
        )
        st.pyplot(fig)

        st.success("XGBoost entrenado correctamente.")

# ---------------------------------------------------------
# BLOQUE 7 — Comparativa de modelos
# ---------------------------------------------------------

if section == "7. Comparativa":
    st.header("7. Comparativa Random Forest vs XGBoost")

    # 1. Inicializamos el estado persistente para la comparativa
    if "comparativa_ejecutada" not in st.session_state:
        st.session_state["comparativa_ejecutada"] = False

    codigo_seccion_7 = """rf_acc = accuracy_score(y_test, rf_preds)
xgb_acc = accuracy_score(y_test_enc, xgb_preds)

plt.bar(['Random Forest', 'XGBoost'], [rf_acc, xgb_acc])
plt.title('Comparativa de Accuracy')
plt.show()

print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_preds):.3f}")
print(f"XGBoost Accuracy:       {accuracy_score(y_test_enc, xgb_preds):.3f}")"""

    codigo_coloreado_7 = colorear_codigo(codigo_seccion_7)

    # Cambiamos st.code por tu componente retro animado
    retro_typewriter_code(codigo_coloreado_7, key="key_comparativa", height=400)

    # 2. Ejecución al pulsar el botón
    if run_button("Comparar modelos"):
        rf_acc = st.session_state.get("rf_acc", None)
        xgb_acc = st.session_state.get("xgb_acc", None)

        # Validación de dependencias
        if rf_acc is None:
            st.error("❌ Debes entrenar el modelo Random Forest antes de comparar.")
        elif xgb_acc is None:
            st.error("❌ Debes entrenar el modelo XGBoost antes de comparar.")
        else:
            # Si ambos modelos existen, creamos el DataFrame de comparación
            df_comp = pd.DataFrame({
                "Modelo": ["Random Forest", "XGBoost"],
                "Accuracy": [rf_acc, xgb_acc]
            })
            
            # Guardamos todo en la sesión para que persista
            st.session_state["comp_df"] = df_comp
            st.session_state["comp_rf_acc"] = rf_acc
            st.session_state["comp_xgb_acc"] = xgb_acc
            st.session_state["comparativa_ejecutada"] = True

    # 3. PERSISTENCIA (AL FINAL): Se renderiza abajo estable si el interruptor está activo
    if st.session_state["comparativa_ejecutada"]:
        rf_acc_p = st.session_state["comp_rf_acc"]
        xgb_acc_p = st.session_state["comp_xgb_acc"]

        # Tabla comparativa
        st.write("### Tabla comparativa")
        st.dataframe(st.session_state["comp_df"])

        # Gráfica comparativa
        st.write("### Gráfica de Accuracy")
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(["Random Forest", "XGBoost"], [rf_acc_p, xgb_acc_p], color=["steelblue", "#ff7f0e"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Comparativa de modelos")
        st.pyplot(fig)

        # Recomendación automática
        st.write("### Mejor modelo")
        if xgb_acc_p > rf_acc_p:
            st.success(f"🤖 XGBoost es el mejor modelo con un accuracy de **{xgb_acc_p:.4f}**")
        elif rf_acc_p > xgb_acc_p:
            st.success(f"🌲 Random Forest es el mejor modelo con un accuracy de **{rf_acc_p:.4f}**")
        else:
            st.info("⚖️ Ambos modelos tienen exactamente el mismo rendimiento.")

# ---------------------------------------------------------
# SECCIÓN 8 — Conclusiones y análisis final
# ---------------------------------------------------------
if section == "8. Conclusiones y análisis final":
    st.header("Conclusiones del análisis")

    # 1. Inicializamos el estado persistente para las conclusiones
    if "conclusiones_ejecutadas" not in st.session_state:
        st.session_state["conclusiones_ejecutadas"] = False

    codigo_seccion_8 = """print("Random Forest Accuracy:", rf_acc)
print("XGBoost Accuracy:", xgb_acc)

if xgb_acc > rf_acc:
    print("XGBoost es el mejor modelo.")
elif rf_acc > xgb_acc:
    print("Random Forest es el mejor modelo.")
else:
    print("Ambos modelos tienen el mismo rendimiento.")"""

    codigo_coloreado_8 = colorear_codigo(codigo_seccion_8)

    # Cambiamos st.code por tu componente retro animado con la altura idónea
    retro_typewriter_code(codigo_coloreado_8, key="key_conclusiones", height=400)

    # 2. Ejecución al pulsar el botón
    if run_button("Mostrar conclusiones"):
        rf_acc = st.session_state.get("rf_acc", None)
        xgb_acc = st.session_state.get("xgb_acc", None)

        # Validación robusta
        if rf_acc is None or xgb_acc is None:
            st.error("❌ Debes entrenar ambos modelos antes de ver las conclusiones.")
        else:
            # Activamos el interruptor de persistencia
            st.session_state["conclusiones_ejecutadas"] = True

    # 3. PERSISTENCIA (AL FINAL): Mantiene el análisis estable en pantalla
    if st.session_state["conclusiones_ejecutadas"]:
        rf_acc_p = st.session_state["rf_acc"]
        xgb_acc_p = st.session_state["xgb_acc"]

        # Resumen numérico
        st.write("### Resumen de métricas")
        st.write(f"- **Random Forest Accuracy:** `{rf_acc_p:.4f}`")
        st.write(f"- **XGBoost Accuracy:** `{xgb_acc_p:.4f}`")

        # Interpretación dinámica
        st.write("### Interpretación de resultados")

        if xgb_acc_p > rf_acc_p:
            st.success("🚀 **XGBoost es el modelo con mejor rendimiento en este dataset.**")
            st.write("""
            XGBoost suele destacar cuando:
            - Hay relaciones no lineales complejas.
            - El dataset tiene ruido o solapamiento leve.
            - Las variables tienen interacciones fuertes.
            - Se necesita un modelo más robusto y con mejor capacidad de generalización.

            En este caso, las iteraciones del boosting han capturado mejor las sutilezas morfológicas de las especies.
            """)
        elif rf_acc_p > xgb_acc_p:
            st.success("🌲 **Random Forest es el modelo con mejor rendimiento en este dataset.**")
            st.write("""
            Random Forest suele destacar cuando:
            - El dataset es pequeño o de tamaño moderado.
            - Las fronteras de decisión de las clases están bien separadas.
            - Se busca un algoritmo muy estable que no dependa críticamente del ajuste de hiperparámetros.

            Aquí, el ensamble por promediado ha sido más que suficiente para clasificar sin sobreajustar.
            """)
        else:
            st.info("⚖️ **Ambos modelos tienen exactamente el mismo rendimiento.**")
            st.write("""
            Esto indica que:
            - El dataset cuenta con fronteras de decisión limpias y fáciles de separar.
            - Ambos modelos capturan óptimamente las relaciones sin necesidad de mayor complejidad.
            - No hay una ventaja clara en el sobreajuste latente o la capacidad de generalización entre arquitecturas.
            """)

        # Recomendaciones finales estructurales
        st.write("### Recomendaciones para producción")
        st.write("""
        - **Random Forest**: Arquitectura más simple, predicción rápida, ideal si buscas **estabilidad y mantenibilidad**.
        - **XGBoost**: Más potente en escalabilidad, óptimo si el volumen de datos crece drásticamente o si buscas rascar la **máxima precisión** posible a costa de un ligero aumento en la complejidad del despliegue.
        """)

        st.success("Conclusiones generadas correctamente.")

# ---------------------------------------------------------
# SECCIÓN 9 — Ficha de Criterio Ético
# ---------------------------------------------------------

if section == "9. Ficha de Criterio Ético":
    st.header("🧭 Criterio Ético y Limitaciones del Modelo")

    with st.expander("📌 Contexto del dataset"):
        st.info("""
El dataset **Palmer Penguins** fue recopilado por la Dra. Kristen Gorman en la Estación Palmer (Antártida).
Incluye **344 observaciones** de **tres especies** de pingüinos en **tres islas** concretas.

Aunque es excelente para fines pedagógicos, presenta limitaciones claras:

- Procede de **una única región geográfica**.  
- Contiene **pocas especies** y **pocas muestras**.  
- Los modelos entrenados con él **no son generalizables** a otras poblaciones o ecosistemas.
""")

    with st.expander("🤖 Riesgos algorítmicos"):
        st.warning("""
Modelos como **Random Forest** y **XGBoost** pueden:

- Sobreajustar fácilmente en datasets pequeños (de hecho, aquí se alcanza **100% accuracy**).  
- Actuar como **cajas negras**, dificultando explicar por qué predicen lo que predicen.  
- Requerir validación adicional (p. ej., **cross-validation**) antes de confiar en sus resultados.
""")

    with st.expander("⚠️ Riesgos si este flujo se aplicara a datos sensibles"):
        st.error("""
Aunque este dataset es inocuo, **el mismo pipeline** aplicado a datos reales podría causar:

- **Sesgos amplificados** por mala limpieza de outliers.  
- **Fugas de información** entre train/test que inflan métricas.  
- **Decisiones injustas** si se confía solo en accuracy.  

Esto puede tener impacto negativo en ámbitos como:
- salud,  
- concesión de crédito,  
- selección de personal,  
- evaluación de riesgo.
""")

    with st.expander("📝 Nota final"):
        st.success("""
Este bloque es una **reflexión ética complementaria** para acompañar la entrega técnica.  
No sustituye un análisis formal de impacto algorítmico.
""")
