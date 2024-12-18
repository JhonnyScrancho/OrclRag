from typing import List
import streamlit as st
from config import INDEX_NAME
import logging

logger = logging.getLogger(__name__)

def ensure_index_exists(pinecone):
    """Connette all'indice Pinecone esistente."""
    try:
        # Get list of existing indexes
        existing_indexes = pinecone.list_indexes()
        
        # Debug: mostra gli indici disponibili nell'interfaccia
        st.write("Debug - Indici disponibili:", existing_indexes)
        logger.info(f"Debug - Indici disponibili: {existing_indexes}")
        logger.info(f"Debug - Tentativo di connessione all'indice: {INDEX_NAME}")
        
        if INDEX_NAME not in existing_indexes:
            st.error(f"""
            L'indice {INDEX_NAME} non esiste tra gli indici disponibili: {existing_indexes}.
            Per favore verifica il nome corretto dell'indice nella console Pinecone.
            """)
            raise ValueError(f"L'indice {INDEX_NAME} non esiste.")
        
        # Get the index
        logger.info(f"Connecting to existing index: {INDEX_NAME}")
        return pinecone.Index(INDEX_NAME)
        
    except Exception as e:
        logger.error(f"Error in ensure_index_exists: {str(e)}", exc_info=True)
        st.error(f"""
        Errore durante la connessione a Pinecone: {str(e)}
        - API Key: {'*' * len(st.secrets['PINECONE_API_KEY'])} (lunghezza: {len(st.secrets['PINECONE_API_KEY'])})
        - Environment: {st.secrets['PINECONE_ENVIRONMENT']}
        - Indice richiesto: {INDEX_NAME}
        - Indici disponibili: {existing_indexes}
        """)
        raise

def update_document_in_index(index, doc_id: str, embedding: List[float], metadata: dict):
    """Aggiorna o inserisce un documento nell'indice."""
    try:
        index.upsert(vectors=[(doc_id, embedding, metadata)])
    except Exception as e:
        logger.error(f"Error in update_document_in_index: {str(e)}", exc_info=True)
        raise