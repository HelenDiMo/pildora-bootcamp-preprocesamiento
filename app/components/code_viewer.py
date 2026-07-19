import inspect
import streamlit as st
import re

def colorear_codigo(codigo_puro: str) -> str:
    """
    Versión ultra-robusta con segmentación de HTML y soporte para azul eléctrico
    en corchetes [] y llaves {}.
    """
    if not codigo_puro:
        return ""

    codigo_normalizado = codigo_puro.replace('\r\n', '\n').replace('\r', '\n')
    lineas = codigo_normalizado.split('\n')
    lineas_procesadas = []
    
    KEYWORDS = ['def', 'import', 'from', 'as', 'return', 'if', 'else', 'elif', 'for', 'in', 'print', 
                'load_dataset', 'head', 'shape', 'isnull', 'sum', 'describe', 'quantile', 'clip', 'show', 'subplots',
                'boxplot', 'title', 'fit', 'predict', 'fit_transform', 'transform', 'bar']
    BUILTINS = ['sns', 'pd', 'np', 'plt', 'X', 'y', 'X_train', 'X_test', 'y_train', 'y_test', 'Pipeline', 'RandomForestClassifier', 
                'ColumnTransformer', 'SimpleImputer', 'OneHotEncoder', 'LabelEncoder', 'XGBClassifier']

    for linea in lineas:
        if not linea.strip():
            lineas_procesadas.append(linea)
            continue
            
        linea = linea.replace('\xa0', ' ')
            
        if linea.lstrip().startswith('#'):
            lineas_procesadas.append(f"<span class='comment'>{linea}</span>")
            continue
            
        # 1. Cadenas de texto primero
        linea = re.sub(r"('[^']*')", r"<span class='string'>\1</span>", linea)
        linea = re.sub(r'("[^"]*")', r"<span class='string'>\1</span>", linea)
        
        # 3. Operadores y símbolos estructurales
        linea = re.sub(r'\(', "<span class='operator'>(</span>", linea)
        linea = re.sub(r'\)', "<span class='operator'>)</span>", linea)
        linea = re.sub(r'\+', "<span class='operator'>+</span>", linea)
        linea = re.sub(r'\-', "<span class='operator'>-</span>", linea)
        linea = re.sub(r'\*', "<span class='operator'>*</span>", linea)
        
        # --- NUEVO: Corchetes y llaves en azul eléctrico ---
        linea = re.sub(r'\[', "<span class='brackets'>[</span>", linea)
        linea = re.sub(r'\]', "<span class='brackets'>]</span>", linea)
        linea = re.sub(r'\{', "<span class='brackets'>{</span>", linea)
        linea = re.sub(r'\}', "<span class='brackets'>}</span>", linea)
        
        # 4. Números enteros y decimales
        linea = re.sub(r'\b(\d+(?:\.\d+)?)\b', r"<span class='number'>\1</span>", linea)

        # 2. KEYWORDS Y BUILTINS: Con segmentación para respetar etiquetas creadas
        partes = re.split(r'(<[^>]+>)', linea)
        
        for i in range(len(partes)):
            if not partes[i].startswith('<'):
                for kw in KEYWORDS:
                    partes[i] = re.sub(r'\b' + kw + r'\b', f"<span class='keyword'>{kw}</span>", partes[i])
                for b in BUILTINS:
                    partes[i] = re.sub(r'\b' + b + r'\b', f"<span class='builtin'>{b}</span>", partes[i])
        
        lineas_procesadas.append("".join(partes))
        
    return '\n'.join(lineas_procesadas)