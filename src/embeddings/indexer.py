# indexer.py

from datetime import datetime, timedelta
import hashlib
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

class PineconeManager:
    def __init__(self, index):
        self.index = index
        self.batch_size = 100
        self.cleanup_threshold_days = 30
        self.namespace = "default"

    def bulk_upsert(self, vectors, batch_size=None):
        """Efficient bulk upsert with batching"""
        if batch_size is None:
            batch_size = self.batch_size
            
        total_vectors = len(vectors)
        progress_bar = st.progress(0)
        
        for i in range(0, total_vectors, batch_size):
            batch = vectors[i:i + batch_size]
            try:
                self.index.upsert(
                    vectors=batch,
                    namespace=self.namespace
                )
                progress = min(1.0, (i + batch_size) / total_vectors)
                progress_bar.progress(progress)
            except Exception as e:
                logger.error(f"Error in bulk upsert batch {i//batch_size}: {str(e)}")
                raise
                
        progress_bar.progress(1.0)
        return total_vectors

    def cleanup_old_vectors(self, days=None):
        """Cleanup vectors older than threshold"""
        if days is None:
            days = self.cleanup_threshold_days
            
        try:
            threshold_date = datetime.now() - timedelta(days=days)
            
            # Query for old vectors
            filter = {
                "timestamp": {"$lt": threshold_date.isoformat()}
            }
            
            # Get vectors to delete
            results = self.index.query(
                vector=[1.0] + [0.0] * (EMBEDDING_DIMENSION-1),
                filter=filter,
                top_k=10000,
                include_metadata=True
            )
            
            if not results.matches:
                return 0
                
            # Delete in batches
            vector_ids = [match.id for match in results.matches]
            total_deleted = 0
            
            for i in range(0, len(vector_ids), self.batch_size):
                batch = vector_ids[i:i + self.batch_size]
                self.index.delete(
                    ids=batch,
                    namespace=self.namespace
                )
                total_deleted += len(batch)
                
            return total_deleted
            
        except Exception as e:
            logger.error(f"Error in cleanup: {str(e)}")
            raise

    def get_index_stats(self):
        """Get cached index statistics"""
        try:
            # Utilizziamo una funzione interna per il caching
            @st.cache_data(ttl=3600)
            def _cached_stats(dimension: int):
                stats = self.index.describe_index_stats()
                return {
                    'total_vectors': stats.total_vector_count,
                    'dimension': stats.dimension,
                    'namespaces': stats.namespaces,
                    'index_fullness': stats.index_fullness
                }
            return _cached_stats(EMBEDDING_DIMENSION)
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return None 