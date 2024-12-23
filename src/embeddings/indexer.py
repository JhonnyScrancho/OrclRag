# indexer.py

import streamlit as st
from config import INDEX_NAME, EMBEDDING_DIMENSION
import logging
from pinecone import Pinecone

logger = logging.getLogger(__name__)

def ensure_index_exists(api_key):
    """Verifica e connette all'indice Pinecone."""
    try:
        logger.info("Initializing Pinecone connection...")
        pc = Pinecone(api_key=api_key)
        
        # Verifica indice esistente
        index = pc.Index(INDEX_NAME)
        
        # Verifica dimensione corretta
        stats = index.describe_index_stats()
        index_dimension = stats.dimension
        
        logger.info(f"Connected to index. Dimension: {index_dimension}")
        
        if index_dimension != EMBEDDING_DIMENSION:
            error_msg = f"Dimensione indice non corretta. Attesa: {EMBEDDING_DIMENSION}, Attuale: {index_dimension}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        return index
        
    except Exception as e:
        error_msg = f"Errore connessione Pinecone: {str(e)}"
        logger.error(error_msg)
        raise

def update_document_in_index(index, doc_id, embedding, metadata):
    """Aggiorna documento nell'indice con verifica dimensione."""
    try:
        # Verifica dimensione embedding
        if len(embedding) != EMBEDDING_DIMENSION:
            raise ValueError(f"Dimensione embedding non valida: {len(embedding)}")
            
        logger.info(f"Updating document {doc_id} in index")
        
        index.upsert(
            vectors=[{
                "id": doc_id,
                "values": embedding,
                "metadata": metadata
            }]
        )
        
        logger.info(f"Successfully updated document {doc_id}")
        
    except Exception as e:
        logger.error(f"Errore aggiornamento documento {doc_id}: {str(e)}")
        raise