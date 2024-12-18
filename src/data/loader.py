import json
from typing import Dict, List, Optional
import streamlit as st
import logging

logger = logging.getLogger(__name__)

def validate_thread(thread: Dict) -> bool:
    """
    Valida la struttura di un thread.
    
    Returns:
        bool: True se il thread è valido, False altrimenti
    """
    required_fields = ['title', 'url', 'scrape_time', 'posts']
    return all(field in thread for field in required_fields)

def load_json(uploaded_file) -> Optional[List[Dict]]:
    """
    Carica e valida il file JSON.
    
    Args:
        uploaded_file: File JSON caricato tramite st.file_uploader
        
    Returns:
        Optional[List[Dict]]: Lista di thread se valida, None altrimenti
    """
    try:
        content = json.load(uploaded_file)
        
        if not isinstance(content, list):
            raise ValueError("Il JSON deve contenere una lista di thread")
            
        # Valida ogni thread
        invalid_threads = []
        valid_threads = []
        
        for i, thread in enumerate(content):
            if validate_thread(thread):
                valid_threads.append(thread)
            else:
                invalid_threads.append(i)
        
        if invalid_threads:
            st.warning(f"Trovati {len(invalid_threads)} thread non validi. Indici: {invalid_threads}")
        
        if not valid_threads:
            raise ValueError("Nessun thread valido trovato nel file")
            
        logger.info(f"Caricati {len(valid_threads)} thread validi")
        return valid_threads
        
    except json.JSONDecodeError as e:
        st.error(f"File JSON non valido: {str(e)}")
        logger.error("Errore di decodifica JSON", exc_info=True)
        return None
    except Exception as e:
        st.error(f"Errore nel caricamento del file: {str(e)}")
        logger.error("Errore nel caricamento del file", exc_info=True)
        return None