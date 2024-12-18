from typing import List
import streamlit as st
from config import INDEX_NAME, VALID_ENVIRONMENTS
import logging
import pinecone

logger = logging.getLogger(__name__)

def validate_environment(environment: str) -> bool:
    """Valida il formato dell'environment di Pinecone."""
    return environment in VALID_ENVIRONMENTS

def ensure_index_exists(pinecone_client):
    """Connette all'indice Pinecone esistente."""
    try:
        environment = st.secrets['PINECONE_ENVIRONMENT']
        if not validate_environment(environment):
            error_msg = f"Environment non valido: {environment}. Formati validi: {', '.join(VALID_ENVIRONMENTS)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Get list of existing indexes
        existing_indexes = pinecone_client.list_indexes()
        
        if not existing_indexes:
            error_msg = "Nessun indice trovato. Creane uno dalla console Pinecone."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if INDEX_NAME not in existing_indexes:
            error_msg = f"L'indice {INDEX_NAME} non esiste. Indici disponibili: {', '.join(existing_indexes)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Get the index
        index = pinecone_client.Index(INDEX_NAME)
        logger.info(f"Connesso con successo all'indice: {INDEX_NAME}")
        return index
        
    except Exception as e:
        error_msg = f"""
        Errore durante la connessione a Pinecone: {str(e)}
        - Environment: {st.secrets['PINECONE_ENVIRONMENT']}
        - Indice richiesto: {INDEX_NAME}
        """
        logger.error(error_msg)
        raise

def update_document_in_index(index, doc_id: str, embedding: List[float], metadata: dict):
    """Aggiorna o inserisce un documento nell'indice."""
    try:
        index.upsert(vectors=[(doc_id, embedding, metadata)])
    except Exception as e:
        logger.error(f"Error in update_document_in_index: {str(e)}", exc_info=True)
        raise