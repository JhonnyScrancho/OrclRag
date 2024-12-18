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
        logger.info(f"Existing indexes: {existing_indexes}")
        
        if INDEX_NAME not in existing_indexes:
            raise ValueError(f"L'indice {INDEX_NAME} non esiste. Crealo prima dalla console Pinecone.")
        
        # Get the index
        logger.info(f"Connecting to existing index: {INDEX_NAME}")
        return pinecone.Index(INDEX_NAME)
        
    except Exception as e:
        logger.error(f"Error in ensure_index_exists: {str(e)}", exc_info=True)
        raise

def update_document_in_index(index, doc_id: str, embedding: List[float], metadata: dict):
    """Aggiorna o inserisce un documento nell'indice."""
    try:
        index.upsert(vectors=[(doc_id, embedding, metadata)])
    except Exception as e:
        logger.error(f"Error in update_document_in_index: {str(e)}", exc_info=True)
        raise