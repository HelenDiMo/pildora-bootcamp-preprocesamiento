import streamlit as st
import time

def run_button(label="Run"):
    return st.button(label)

def stream_code(code_text):
    """Simula escritura de código línea a línea."""
    for char in code_text:
        yield char
        time.sleep(0.02)