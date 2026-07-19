import streamlit as st
import streamlit.components.v1 as components

def retro_typewriter_code(html_content: str, key: str, height: int = 250):
    """
    Renderiza un bloque de código con efecto máquina de escribir retro persistente.
    Se anima estrictamente al pulsar '▶ RUN CODE' y se mantiene fijo en futuros reruns.
    """
    # 1. Inicializar el estado de visibilidad en Streamlit
    if key not in st.session_state:
        st.session_state[key] = False

    # 2. Cargar el CSS personalizado
    try:
        with open("app/assets/styles.css") as f:
            custom_css = f.read()
    except FileNotFoundError:
        try:
            with open("assets/styles.css") as f:
                custom_css = f.read()
        except FileNotFoundError:
            custom_css = ""

    # Botón nativo de Streamlit
    if st.button("▶ RUN CODE", key=f"btn_{key}"):
        st.session_state[key] = True
        # Usamos un pequeño truco inyectando un script temporal para limpiar el almacenamiento local
        # Esto fuerza a que este bloque específico SE SIENTA obligado a animarse de nuevo
        components.html(f"<script>window.localStorage.removeItem('{key}_done');</script>", height=0)

    # Si nunca se ha pulsado el botón en esta sesión, mantenemos el contenedor invisible o vacío
    #if not st.session_state[key]:
        #components.html("", height=0)
        #return

    html_code = f"""
    <style>
        {custom_css}
        body {{
            margin: 0;
            padding: 0;
            background-color: transparent;
            overflow: hidden;
        }}
        pre {{
            margin-top: 5px !important;
            margin-bottom: 0px !important;
        }}
    </style>

    <pre><code id="typewriter"></code></pre>

    <script>
        const rawHtml = {repr(html_content)};
        const storageKey = '{key}_done';
        const target = document.getElementById('typewriter');
        
        // Comprobamos si se pulsó el botón en Streamlit
        const isActivated = { "true" if st.session_state[key] else "false" };
        const hasAnimatedBefore = window.localStorage.getItem(storageKey);

        // 1. PRIORIDAD: Si el botón NO se ha pulsado en Streamlit, forzamos contenedor VACÍO
        if (!isActivated) {{
            target.innerHTML = "";
        }} 
        // 2. Si el botón SÍ está pulsado y ya se animó antes, se queda fijo de golpe
        else if (hasAnimatedBefore === 'true') {{
            target.innerHTML = rawHtml;
        }} 
        // 3. Si el botón SÍ está pulsado pero es la primera vez, se ejecuta el efecto
        else {{
            let i = 0;
            let currentText = "";
            
            function type() {{
                if (i < rawHtml.length) {{
                    if (rawHtml[i] === '<') {{
                        let closingTagIndex = rawHtml.indexOf('>', i);
                        currentText += rawHtml.substring(i, closingTagIndex + 1);
                        i = closingTagIndex + 1;
                    }} else {{
                        currentText += rawHtml[i];
                        i++;
                    }}
                    target.innerHTML = currentText;
                    setTimeout(type, 15);
                }} else {{
                    window.localStorage.setItem(storageKey, 'true');
                }}
            }}
            type();
        }}
    </script>
    """
    
    components.html(html_code, height=height)

    # Si JS nos avisa de que terminó de escribir, actualizamos el Session State de Streamlit
    #if response:
    #    st.session_state[key] = True
    #    st.rerun()

def run_button(label="Run"):
    return st.button(label)



