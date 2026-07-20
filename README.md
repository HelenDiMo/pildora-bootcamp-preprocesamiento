# 🧠 Masterclass / Píldora Extendida: Preprocesamiento Avanzado y Algoritmos Ensemble

Sesión MasterClass de estilo "píldora extendida" para el **Modulo 3 - Machine Learning** sobre **preprocesamiento avanzado** (outliers, encoding, feature engineering) y **algoritmos ensemble** (Random Forest, XGBoost, Bagging vs. Boosting), usando el dataset **Palmer Penguins**.

## 📋 Índice
 
- [Contenido de la entrega](#-contenido-de-la-entrega)
- [Datasets utilizados](#-datasets-utilizados)
- [Requisitos e instalación](#️-requisitos-e-instalación)
- [Cómo ejecutar cada parte](#️-cómo-ejecutar-cada-parte)
- [Recursos externos](#-recursos-externos)
- [Primeros auxilios (Troubleshooting)](#-primeros-auxilios-troubleshooting)
- [Criterio ético](#-criterio-ético)
- [Autoría](#-autoría)
---

## 📦 Contenido de la entrega

| Entregable | Enlace / archivo |
| --- | --- |
| Presentación teórica (Genially) | [https://view.genially.com/6a500ced927e47fbbe0a9fea](https://view.genially.com/6a500ced927e47fbbe0a9fea) |
| Presentación teórica (`.pdf`) | `presentacion_pptx_pdf.pdf` |
| Demo interactiva (Streamlit) | [https://bootcamp-preprocesamiento.streamlit.app/](https://bootcamp-preprocesamiento.streamlit.app/) |
| Notebook con la solución completa (live coding) | `live_coding_demo.ipnb` |
| Reto de Aplicación - Para los compañeros | `reto_notebook_bloque3_tips_v2.ipynb` |
| Reto de Aplicación - _SOLUCIÓN_ | `reto_notebook_bloque3_tips_SOLUCION_v2.ipynb` |
| Documentación | [https://mintlify.wiki/HelenDiMo/pildora-bootcamp-preprocesamiento/introduction](https://mintlify.wiki/HelenDiMo/pildora-bootcamp-preprocesamiento/introduction) |

---
 
## 📊 Datasets utilizados
 
Se usan **dos datasets distintos a propósito**, uno por bloque:
 
| Dataset | Dónde se usa | Por qué |
|---|---|---|
| **Palmer Penguins** (`sns.load_dataset('penguins')`) | Live coding | Dataset limpio y sencillo, ideal para explicar el flujo paso a paso sin distracciones. Suele dar accuracy muy alto (incluso 1.0), lo cual es útil para introducir el debate sobre sobreajuste |
| **Tips** (`sns.load_dataset('tips')`) | Reto de aplicación | Da resultados más realistas y variados (accuracy ~0.83-0.88, con precision/recall claramente distintos entre clases por el desbalance Lunch/Dinner) — mejor material para el debate del Bloque 4 |
 
---

## ⚙️ Requisitos e instalación

```bash
pip install pandas seaborn scikit-learn xgboost matplotlib
```

Si vais a abrir el notebook del reto en **Google Colab**, no hace falta instalar nada salvo XGBoost:

```python
!pip install xgboost -q
```

---

## ▶️ Cómo ejecutar cada parte

1. **Live Coding:** Abrid `live_coding_demo.ipynb` en VSCode, Jupyter o Google Colab: es el notebook de la demo.
2. **Reto:** Abrid `reto_notebook_bloque3_tips_v2.ipynb` en VSCode, Jupyter o Google Colab y completad los bloques marcados con `# TODO: Implementar aquí`.

---
## 🔗 Recursos externos
 
- **Presentación teórica (Genially):** https://view.genially.com/6a500ced927e47fbbe0a9fea - Presentación en diapositivas interactivas con aclaraciones y seccion de *Recursos Recomendados* por conceptos.

- **Demo interactiva (Streamlit):** https://bootcamp-preprocesamiento.streamlit.app/ - Demo interactiva estilo retro que muestra el código trabajado en `live_coding_demo.ipynb`
---


## 🚑 Primeros auxilios (Troubleshooting)
 
| Error | Causa habitual | Solución |
|---|---|---|
| `ModuleNotFoundError: No module named 'xgboost'` | XGBoost no está instalado en tu entorno | `pip install xgboost` (o `!pip install xgboost -q` en Colab) |
| `NameError: name 'X_train' is not defined` | Se ha ejecutado una celda sin ejecutar antes las anteriores | Ejecutad las celdas **en orden**, de arriba a abajo (Run All si dudáis) |
| Al re-ejecutar la celda de outliers, el número de outliers detectados cambia (o da 0) | La celda calcula `Q1`/`Q3` sobre `df`, pero esa misma celda ya modificó `df` la vez anterior (con `.clip()`) — la segunda ejecución mide sobre datos ya "aplanados" | Guardad una copia intacta del dataset original nada más cargarlo (`df_raw = df.copy()`) y calculad siempre `Q1`/`Q3`/outliers sobre `df_raw`, no sobre `df` |
| `sns.load_dataset(...)` falla o se queda colgado | Sin conexión a internet, o firewall/proxy bloqueando GitHub | Comprobad vuestra conexión; si seguís sin acceso, pedid el CSV local como alternativa |
| `ValueError: Input contains NaN` al entrenar el modelo | Se ha intentado entrenar el modelo directamente, saltándose el `preprocessor`/pipeline | Aseguraos de entrenar siempre con el pipeline completo (`rf_pipeline`, `xgb_pipeline`...), no con el modelo suelto |
| `ValueError` por categorías desconocidas en test | Al `OneHotEncoder` le falta `handle_unknown='ignore'` | Revisad que el encoder tenga ese parámetro configurado |
| `KeyError: 'columna_x'` | Typo en el nombre de columna, o columna no incluida en `num_cols`/`cat_cols` | Comprobad `df.columns` y que el nombre coincide exactamente (mayúsculas, guiones bajos, etc.) |
| Al entrenar XGBoost dos veces con el mismo `preprocessor`, el segundo modelo "pisa" al primero | Se reutiliza la misma instancia de `ColumnTransformer` ya ajustada en dos pipelines distintos | Usad `from sklearn.base import clone` y pasad `clone(preprocessor)` al segundo pipeline |
| Accuracy = 1.0 y os genera dudas | Dataset pequeño y fácilmente separable (típico en Penguins), o posible sobreajuste | Validad con `cross_val_score(pipeline, X, y, cv=5)` para confirmar que no es casualidad del split |
| `ModuleNotFoundError` al importar `seaborn`/`sklearn` | Entorno virtual incorrecto o librería no instalada | `pip install pandas seaborn scikit-learn xgboost matplotlib` en el entorno activo |

---

## 🧭 Criterio ético
 
Ver Sección 9 de [Demo interactiva (Streamlit)](https://bootcamp-preprocesamiento.streamlit.app/) — Donde se resumen los riesgos de sesgo, las limitaciones de los algoritmos usados y el impacto potencial de aplicar este mismo flujo de trabajo a datos sensibles.
 
---

## 👥 Autoría

Preparado por Elena Díaz - [@HelenDiMo](https://github.com/HelenDiMo) para el bloque de Machine Learning de Bootcamp IA & Data Science de Somos F5