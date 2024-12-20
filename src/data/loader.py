import json
from typing import Dict, List
import streamlit as st

def load_json(uploaded_file) -> List[Dict]:
    """Carica e valida il file JSON."""
    try:
        content = json.load(uploaded_file)
        if not isinstance(content, list):
            raise ValueError("Il JSON deve contenere una lista di thread")
        return content
    except json.JSONDecodeError:
        st.error("File JSON non valido")
        return None
    except Exception as e:
        st.error(f"Errore nel caricamento del file: {str(e)}")
        return None
