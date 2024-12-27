import streamlit as st
from datetime import datetime, timedelta
import hashlib
from config import INDEX_NAME, EMBEDDING_DIMENSION
import logging
from pinecone import Pinecone
from typing import Dict, List, Optional, Tuple, Union
import time
import numpy as np

logger = logging.getLogger(__name__)

class PineconeManager:
    def __init__(self, index):
        self.index = index
        self.batch_size = 100
        self.cleanup_threshold_days = 30
        self.namespace = "default"
        self._cache = {
            'last_cleanup': None,
            'query_count': 0,
            'error_count': 0
        }

    def bulk_upsert(self, vectors: List[Dict], batch_size: Optional[int] = None) -> int:
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

    def cleanup_old_vectors(self, days: Optional[int] = None) -> int:
        """Cleanup vectors older than threshold"""
        if days is None:
            days = self.cleanup_threshold_days
            
        try:
            threshold_date = datetime.now() - timedelta(days=days)
            
            # Query for old vectors
            filter_dict = {
                "timestamp": {"$lt": threshold_date.isoformat()}
            }
            
            # Get vectors to delete
            results = self.index.query(
                vector=[1.0] + [0.0] * (EMBEDDING_DIMENSION-1),
                filter=filter_dict,
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

    def get_index_stats(self) -> Optional[Dict]:
        """Get cached index statistics"""
        try:
            # Utilizziamo una funzione interna per il caching
            @st.cache_data(ttl=3600)
            def _cached_stats(dimension: int) -> Dict:
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

    def verify_permissions(self) -> Tuple[bool, str]:
        """Verify delete permissions on the index"""
        try:
            # Create a test vector
            test_id = "test_permissions"
            test_vector = [0.0] * EMBEDDING_DIMENSION
            test_vector[0] = 1.0
            
            # Try insert
            self.index.upsert(
                vectors=[{
                    "id": test_id,
                    "values": test_vector,
                    "metadata": {"test": True}
                }]
            )
            
            time.sleep(0.5)  # Small delay
            
            # Try query
            results = self.index.query(
                vector=test_vector,
                top_k=1,
                include_metadata=True
            )
            
            # Try delete
            self.index.delete(ids=[test_id])
            
            time.sleep(0.5)  # Small delay
            
            # Verify deletion
            verify = self.index.fetch(ids=[test_id])
            if verify and hasattr(verify, 'vectors') and verify.vectors:
                return False, "Insufficient delete permissions"
                
            return True, "Permissions verified successfully"
            
        except Exception as e:
            return False, f"Error verifying permissions: {str(e)}"

    def find_duplicates(self) -> List[Dict]:
        """Find duplicate vectors in the index"""
        try:
            results = self.index.query(
                vector=[0.0] * EMBEDDING_DIMENSION,
                top_k=10000,
                include_metadata=True
            )
            
            if not results.matches:
                return []
                
            # Map for finding duplicates
            content_map = {}
            duplicates = []
            
            for match in results.matches:
                content_key = f"{match.metadata.get('thread_id')}_{match.metadata.get('post_id')}"
                
                if content_key in content_map:
                    duplicates.append({
                        'id': match.id,
                        'content_key': content_key,
                        'metadata': match.metadata
                    })
                else:
                    content_map[content_key] = match.id
                    
            return duplicates
            
        except Exception as e:
            logger.error(f"Error finding duplicates: {str(e)}")
            return []

    def delete_duplicates(self) -> Tuple[int, List[str]]:
        """Delete duplicate vectors"""
        try:
            duplicates = self.find_duplicates()
            if not duplicates:
                return 0, []
                
            deleted_ids = []
            for dup in duplicates:
                try:
                    self.index.delete(ids=[dup['id']])
                    deleted_ids.append(dup['id'])
                except Exception as e:
                    logger.error(f"Error deleting duplicate {dup['id']}: {str(e)}")
                    
            return len(deleted_ids), deleted_ids
            
        except Exception as e:
            logger.error(f"Error in delete_duplicates: {str(e)}")
            return 0, []

    def should_run_cleanup(self) -> bool:
        """Determine if cleanup should run"""
        if not self._cache['last_cleanup']:
            return True
            
        # Cleanup every 24h or 1000 queries
        time_threshold = datetime.now() - timedelta(hours=24)
        return (
            self._cache['last_cleanup'] < time_threshold or
            self._cache['query_count'] >= 1000 or
            self._cache['error_count'] >= 10
        )

def initialize_pinecone():
    """Initialize and verify Pinecone connection"""
    try:
        # Verify API key
        if "PINECONE_API_KEY" not in st.secrets:
            st.error("🔑 Pinecone API key non trovata nelle secrets")
            return None
        
        # Initialize connection
        pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        
        # Get index
        try:
            index = pc.Index(INDEX_NAME)
            
            # Verify index state
            stats = index.describe_index_stats()
            
            # Verify dimension
            index_dimension = stats.dimension
            if index_dimension != EMBEDDING_DIMENSION:
                st.error(f"⚠️ Dimensione indice non corretta. Attesa: {EMBEDDING_DIMENSION}, Attuale: {index_dimension}")
                return None
                
            # Check if index is empty
            if stats.total_vector_count == 0:
                st.warning("📝 Il database è vuoto. Carica dei dati dalla tab 'Database'.")
            
            return index
            
        except Exception as e:
            st.error(f"❌ Errore verifica indice: {str(e)}")
            logger.error(f"Index verification error: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"❌ Errore connessione Pinecone: {str(e)}")
        logger.error(f"Pinecone connection error: {str(e)}")
        return None

def initialize_pinecone_and_manager():
    """Initialize both Pinecone and PineconeManager"""
    index = initialize_pinecone()
    if index is None:
        return None, None
        
    try:
        manager = PineconeManager(index)
        return index, manager
    except Exception as e:
        st.error(f"❌ Errore inizializzazione manager: {str(e)}")
        logger.error(f"Manager initialization error: {str(e)}")
        return None, None