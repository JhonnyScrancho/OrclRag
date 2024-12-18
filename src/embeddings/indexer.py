from typing import List
import streamlit as st
from config import INDEX_NAME
import logging

logger = logging.getLogger(__name__)

def ensure_index_exists(pinecone):
    """Connette all'indice Pinecone esistente."""
    try:
        # Debug: mostra le impostazioni di connessione
        st.write("Tentativo di connessione con:")
        st.write(f"- Indice richiesto: {INDEX_NAME}")
        st.write(f"- Environment: {st.secrets['PINECONE_ENVIRONMENT']}")
        
        # Get list of existing indexes
        existing_indexes = pinecone.list_indexes()
        
        # Debug: mostra gli indici disponibili
        st.write("Indici disponibili:", existing_indexes)
        logger.info(f"Debug - Indici disponibili: {existing_indexes}")
        
        if not existing_indexes:
            st.warning("Nessun indice trovato nel tuo account Pinecone.")
            raise ValueError("Nessun indice disponibile")
            
        if INDEX_NAME not in existing_indexes:
            available_indexes = ', '.join(existing_indexes) if existing_indexes else 'nessuno'
            st.error(f"L'indice {INDEX_NAME} non esiste. Indici disponibili: {available_indexes}")
            raise ValueError(f"L'indice {INDEX_NAME} non esiste")
        
        # Get the index
        logger.info(f"Connecting to existing index: {INDEX_NAME}")
        return pinecone.Index(INDEX_NAME)
        
    except Exception as e:
        error_msg = f"""
        Errore durante la connessione a Pinecone: {str(e)}
        - API Key length: {len(st.secrets['PINECONE_API_KEY'])}
        - Environment: {st.secrets['PINECONE_ENVIRONMENT']}
        - Indice richiesto: {INDEX_NAME}
        """
        logger.error(error_msg)
        st.error(error_msg)
        raise

def update_document_in_index(index, doc_id: str, embedding: List[float], metadata: dict):
    """Aggiorna o inserisce un documento nell'indice."""
    try:
        index.upsert(vectors=[(doc_id, embedding, metadata)])
    except Exception as e:
        logger.error(f"Error in update_document_in_index: {str(e)}", exc_info=True)
        raise