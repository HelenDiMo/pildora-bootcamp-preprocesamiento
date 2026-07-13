import inspect
import streamlit as st

def show_code(func):
    code = inspect.getsource(func)
    st.code(code)
