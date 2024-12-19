from typing import List
import streamlit as st
from config import INDEX_NAME
import logging
import pinecone

logger = logging.getLogger(__name__)

def ensure_index_exists(pinecone_client):
    """Connette all'indice Pinecone esistente."""
    try:
        # Debug: mostra le impostazioni di connessione
        st.write("Tentativo di connessione a Pinecone:")
        st.write(f"- Nome indice richiesto: {INDEX_NAME}")
        
        # Nuova sintassi per Pinecone v3
        pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        
        # Get list of existing indexes
        existing_indexes = pc.list_indexes()
        
        if INDEX_NAME not in existing_indexes:
            available_indexes = ', '.join(existing_indexes) if existing_indexes else 'nessuno'
            error_msg = f"L'indice {INDEX_NAME} non esiste. Per favore, crealo dalla console di Pinecone. Indici disponibili: {available_indexes}"
            st.error(error_msg)
            raise ValueError(error_msg)
        
        # Get the index
        index = pc.Index(INDEX_NAME)
        st.success(f"Connesso con successo all'indice: {INDEX_NAME}")
        return index
        
    except Exception as e:
        error_msg = f"""
        Errore durante la connessione a Pinecone: {str(e)}
        - API Key length: {len(st.secrets['PINECONE_API_KEY'])}
        - Indice richiesto: {INDEX_NAME}
        """
        logger.error(error_msg)
        st.error(error_msg)
        raise
        
    except Exception as e:
        error_msg = f"""
        Errore durante la connessione a Pinecone: {str(e)}
        - API Key length: {len(st.secrets['PINECONE_API_KEY'])}
        - Indice richiesto: {INDEX_NAME}
        """
        logger.error(error_msg)
        st.error(error_msg)
        raise

def update_document_in_index(index, doc_id: str, embedding: List[float], metadata: dict):
    """Aggiorna o inserisce un documento nell'indice."""
    try:
        index.upsert(
            vectors=[{
                "id": doc_id,
                "values": embedding,
                "metadata": metadata
            }]
        )
    except Exception as e:
        logger.error(f"Error in update_document_in_index: {str(e)}", exc_info=True)
        raise