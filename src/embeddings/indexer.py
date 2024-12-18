from typing import List
import streamlit as st
from config import INDEX_NAME
import logging

logger = logging.getLogger(__name__)

def ensure_index_exists(pinecone):
    """Assicura che l'indice Pinecone esista."""
    try:
        # Get list of existing indexes
        existing_indexes = pinecone.list_indexes()
        logger.info(f"Existing indexes: {existing_indexes}")
        
        # Create index if it doesn't exist
        if INDEX_NAME not in existing_indexes:
            logger.info(f"Creating new serverless index: {INDEX_NAME}")
            pinecone.create_index(
                name=INDEX_NAME,
                dimension=1536,
                metric="cosine",
                serverless=True,
                cloud="aws",
                region="us-east-1"  # Usa la stessa regione dell'indice esistente
            )
            logger.info(f"Successfully created serverless index: {INDEX_NAME}")
        else:
            logger.info(f"Index {INDEX_NAME} already exists")
        
        # Get the index
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