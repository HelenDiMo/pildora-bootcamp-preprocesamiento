import streamlit as st
import streamlit.components.v1 as components

def retro_typewriter_code(html_coloreado: str, key: str, height: int = 250):
    """
    Renderiza el código con efecto máquina de escribir mostrando los colores en tiempo real.
    Utiliza un clonador de nodos para evitar roturas por culpa de etiquetas HTML.
    """
    if key not in st.session_state:
        st.session_state[key] = False

    try:
        with open("app/assets/styles.css") as f:
            custom_css = f.read()
    except FileNotFoundError:
        try:
            with open("assets/styles.css") as f:
                custom_css = f.read()
        except FileNotFoundError:
            custom_css = ""

    if st.button("▶ RUN CODE", key=f"btn_{key}"):
        st.session_state[key] = True
        components.html(f"<script>window.localStorage.removeItem('{key}_done');</script>", height=0)

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
        pre code {{
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
            padding: 0 !important;
            display: block;
        }}
        /* Estilo para ocultar el molde original mientras se copia */
        #molde_oculto {{
            display: none;
        }}
    </style>

    <!-- Contenedor real donde escribe la máquina -->
    <pre><code id="typewriter"></code></pre>
    
    <!-- Contenedor oculto con el código ya coloreado por Python -->
    <div id="molde_oculto">{html_coloreado}</div>

    <script>
        const storageKey = '{key}_done';
        const target = document.getElementById('typewriter');
        const molde = document.getElementById('molde_oculto');
        
        const isActivated = { "true" if st.session_state[key] else "false" };
        const hasAnimatedBefore = window.localStorage.getItem(storageKey);

        if (!isActivated) {{
            target.innerHTML = "";
        }} 
        else if (hasAnimatedBefore === 'true') {{
            target.innerHTML = molde.innerHTML;
        }} 
        else {{
            // Conseguimos el HTML estructurado y lo troceamos de forma inteligente
            const htmlCompleto = molde.innerHTML;
            let i = 0;
            let currentHTML = "";
            
            // Este array guardará las posiciones de control para saber cuándo abrir/cerrar etiquetas
            function type() {{
                if (i < htmlCompleto.length) {{
                    // Si el carácter actual es una apertura de etiqueta HTML, la metemos entera de golpe
                    if (htmlCompleto[i] === '<') {{
                        let finEtiqueta = htmlCompleto.indexOf('>', i);
                        currentHTML += htmlCompleto.substring(i, finEtiqueta + 1);
                        i = finEtiqueta + 1;
                        // Ejecutamos recursivamente de inmediato para no meter pausas en las etiquetas
                        type(); 
                    }} else {{
                        // Si es un carácter visible, lo añadimos y esperamos el timeout
                        currentHTML += htmlCompleto[i];
                        target.innerHTML = currentHTML;
                        i++;
                        setTimeout(type, 40); 
                    }}
                }} else {{
                    window.localStorage.setItem(storageKey, 'true');
                }}
            }}
            type();
        }}
    </script>
    """
    components.html(html_code, height=height)

def run_button(label="Run"):
    return st.button(label)



