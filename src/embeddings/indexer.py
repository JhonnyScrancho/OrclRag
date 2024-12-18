from typing import List
import streamlit as st
from config import INDEX_NAME
import logging
import pinecone
import time

logger = logging.getLogger(__name__)

def ensure_index_exists(pinecone_client):
    """Connette all'indice Pinecone esistente o lo crea se non esiste."""
    try:
        # Debug: mostra le impostazioni di connessione
        st.write("Tentativo di connessione a Pinecone:")
        st.write(f"- Nome indice richiesto: {INDEX_NAME}")
        st.write(f"- Environment: {st.secrets['PINECONE_ENVIRONMENT']}")
        
        # Get list of existing indexes
        existing_indexes = pinecone_client.list_indexes()
        
        if not existing_indexes:
            st.warning("Nessun indice trovato. Creo un nuovo indice serverless...")
            # Create a new serverless index
            pinecone_client.create_index(
                name=INDEX_NAME,
                dimension=1536,  # dimensione per OpenAI ada-002
                metric='cosine',
                service_type='serverless',
                cloud='aws',
                region='us-east-1'
            )
            # Wait for index to be ready
            time.sleep(1)
            while not pinecone_client.describe_index(INDEX_NAME).status['ready']:
                time.sleep(1)
            
        elif INDEX_NAME not in existing_indexes:
            st.warning(f"L'indice {INDEX_NAME} non esiste. Creo un nuovo indice serverless...")
            pinecone_client.create_index(
                name=INDEX_NAME,
                dimension=1536,
                metric='cosine',
                service_type='serverless',
                cloud='aws',
                region='us-east-1'
            )
            # Wait for index to be ready
            time.sleep(1)
            while not pinecone_client.describe_index(INDEX_NAME).status['ready']:
                time.sleep(1)
        
        # Get the index
        index = pinecone_client.Index(INDEX_NAME)
        st.success(f"Connesso con successo all'indice: {INDEX_NAME}")
        return index
        
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