import streamlit as st
from data.loader import load_json

def render_sidebar():
    """Renderizza la sidebar con le funzionalità di upload."""
    with st.sidebar:
        st.header("Caricamento Dati")
        uploaded_file = st.file_uploader("Carica JSON del forum", type=['json'])
        
        process_button = st.button("Processa")
        
        if uploaded_file and process_button:
            data = load_json(uploaded_file)
            if data:
                return data
        
        return None