# Masterclass / Píldora Extendida: Preprocesamiento Avanzado y Algoritmos Ensemble

Sesión de la "píldora extendida" sobre preprocesamiento avanzado (outliers, encoding, feature engineering) y algoritmos ensemble (Random Forest, XGBoost, Bagging vs. Boosting), usando el dataset **Palmer Penguins**.

---

## 📦 Contenido de la entrega

| Entregable | Enlace / archivo |
|---|---|
| Presentación teórica (Genially) | https://view.genially.com/6a500ced927e47fbbe0a9fea |
| Demo interactiva (Streamlit) | https://bootcamp-preprocesamiento.streamlit.app/ |
| Notebook con la solución completa (live coding) | `live_coding_demo.ipnb` |
| Notebook con TODOs — reto de aplicación | `reto_notebook_bloque3.ipynb` |
| Ficha de Criterio Ético | `ficha_criterio_etico.md` |

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

## ▶️ Cómo ejecutar

1. Abrid `live_coding_demo.ipnb` en VSCode, es el live coding de la demo. 
2. Para el reto, abrid `reto_notebook_bloque3.ipynb` en Jupyter o Google Colab y completad los bloques marcados con `# TODO: Implementar aquí`.

---

## 🚑 Primeros auxilios (Troubleshooting)

| Error | Causa habitual | Solución |
|---|---|---|
| `ModuleNotFoundError: No module named 'xgboost'` | XGBoost no está instalado en tu entorno | `pip install xgboost` (o `!pip install xgboost -q` en Colab) |
| `NameError: name 'X_train' is not defined` | Se ha ejecutado una celda sin ejecutar antes las anteriores | Ejecutad las celdas **en orden**, de arriba a abajo (Run All si dudáis) |
| `seaborn.load_dataset('penguins')` falla o se queda colgado | Sin conexión a internet, o firewall/proxy bloqueando GitHub | Comprobad vuestra conexión; si seguís sin acceso, pedid el CSV local como alternativa |
| `ValueError: Input contains NaN` al entrenar el modelo | Se ha intentado entrenar el modelo directamente, saltándose el `preprocessor`/pipeline | Aseguraos de entrenar siempre con `rf_pipeline` o `xgb_pipeline` completo, no con el modelo suelto |
| `ValueError` por categorías desconocidas en test | Al `OneHotEncoder` le falta `handle_unknown='ignore'` | Revisad que el encoder tenga ese parámetro configurado |
| `KeyError: 'columna_x'` | Typo en el nombre de columna, o columna no incluida en `num_cols`/`cat_cols` | Comprobad `df.columns` y que el nombre coincide exactamente (mayúsculas, guiones bajos, etc.) |
| Accuracy = 1.0 y os genera dudas | Dataset pequeño y fácilmente separable, o posible sobreajuste | Validad con `cross_val_score(rf_pipeline, X, y, cv=5)` para confirmar que no es casualidad del split |
| `ModuleNotFoundError` al importar `seaborn`/`sklearn` | Entorno virtual incorrecto o librería no instalada | `pip install pandas seaborn scikit-learn xgboost matplotlib` en el entorno activo |

---

## 👥 Autoría
Preparado por Elena Díaz - [@HelenDiMo](https://github.com/HelenDiMo) para el bloque de Machine Learning del bootcamp.
