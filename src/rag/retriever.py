from sentence_transformers import CrossEncoder
import streamlit as st
from typing import List, Dict, Any
from langchain_core.documents import Document
import logging
from datetime import datetime
from config import EMBEDDING_DIMENSION, INITIAL_RETRIEVAL_K, FINAL_K
from functools import lru_cache

logger = logging.getLogger(__name__)

class SmartRetriever:
    def __init__(self, index, embeddings):
        self.index = index
        self.embeddings = embeddings
        self.MAX_DOCUMENTS = 1000  # Aumentato significativamente
        self.FINAL_K = FINAL_K
        self.EMBEDDING_DIMENSION = EMBEDDING_DIMENSION
        # Initialize cross-encoder for re-ranking
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    @lru_cache(maxsize=100)
    def _cached_documents(self, query_vector_key: tuple, top_k: int) -> List[Any]:
        """Internal cached function for document retrieval"""
        try:
            # Convert back to list for Pinecone
            query_vector = list(query_vector_key)
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )
            return results.matches if results and hasattr(results, 'matches') else []
        except Exception as e:
            logger.error(f"Error in cached document retrieval: {str(e)}")
            return []

    def _fetch_thread_documents(self, thread_id: str) -> List[Any]:
        """Fetch all documents for a specific thread."""
        try:
            results = self.index.query(
                vector=[1.0] + [0.0] * (self.EMBEDDING_DIMENSION - 1),
                filter={"thread_id": thread_id},
                top_k=self.MAX_DOCUMENTS,
                include_metadata=True
            )
            return results.matches if results and hasattr(results, 'matches') else []
        except Exception as e:
            logger.error(f"Error fetching thread documents: {str(e)}")
            return []

    def _rerank_documents(self, query: str, documents: List[Document], top_k: int = None) -> List[Document]:
        """Re-rank documents using cross-encoder"""
        if not documents:
            return []
            
        if top_k is None:
            top_k = self.FINAL_K

        try:
            # Prepare pairs for cross-encoder
            pairs = [[query, doc.page_content] for doc in documents]
            
            # Get cross-encoder scores
            scores = self.cross_encoder.predict(pairs)
            
            # Create document-score pairs and sort
            doc_score_pairs = list(zip(documents, scores))
            reranked_docs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
            
            # Return top-k documents
            return [doc for doc, _ in reranked_docs[:top_k]]
            
        except Exception as e:
            logger.error(f"Error in document re-ranking: {str(e)}")
            return documents[:top_k]  # Fallback to original order

    def get_all_documents(self) -> List[Any]:
        """Retrieve all documents using zero vector with first element as 1"""
        try:
            query_vector = [0.0] * self.EMBEDDING_DIMENSION
            query_vector[0] = 1.0
            
            # Use Streamlit's cache_data at function call level
            @st.cache_data(ttl=3600)
            def get_cached_docs(vector_key: tuple):
                return self._cached_documents(vector_key, self.MAX_DOCUMENTS)
            
            return get_cached_docs(tuple(query_vector))
            
        except Exception as e:
            logger.error(f"Error fetching all documents: {str(e)}")
            return []

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Enhanced document retrieval with complete thread fetching."""
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            if len(query_embedding) != self.EMBEDDING_DIMENSION:
                raise ValueError(f"Query embedding dimension mismatch: {len(query_embedding)} vs {self.EMBEDDING_DIMENSION}")
            
            # Initial retrieval with high limit
            results = self.index.query(
                vector=query_embedding,
                top_k=INITIAL_RETRIEVAL_K,  # Get more initial results
                include_metadata=True
            )

            if not results.matches:
                logger.warning("No initial matches found")
                return []

            # Collect unique thread IDs
            thread_ids = set(doc.metadata.get("thread_id") for doc in results.matches if doc.metadata)
            logger.info(f"Found {len(thread_ids)} unique threads")

            # Fetch complete threads
            all_documents = []
            for thread_id in thread_ids:
                thread_docs = self._fetch_thread_documents(thread_id)
                logger.info(f"Thread {thread_id}: found {len(thread_docs)} documents")
                all_documents.extend(thread_docs)

            # Convert to Document objects
            documents = []
            for doc in all_documents:
                if not doc.metadata.get("text"):
                    logger.warning(f"Document missing text: {doc.id}")
                    continue
                    
                documents.append(Document(
                    page_content=doc.metadata.get("text", ""),
                    metadata=self._format_metadata(doc.metadata)
                ))

            # Log retrieval statistics
            logger.info(f"Total documents retrieved: {len(documents)}")
            thread_stats = {}
            for doc in documents:
                thread_id = doc.metadata.get("thread_id")
                if thread_id not in thread_stats:
                    thread_stats[thread_id] = {
                        "count": 0,
                        "total": doc.metadata.get("total_posts", 0)
                    }
                thread_stats[thread_id]["count"] += 1

            for thread_id, stats in thread_stats.items():
                logger.info(f"Thread {thread_id}: {stats['count']}/{stats['total']} posts")

            # Re-rank if we have more documents than needed
            if len(documents) > self.FINAL_K:
                documents = self._rerank_documents(query, documents)

            # Sort by timestamp if available
            try:
                documents.sort(key=lambda x: datetime.strptime(
                    x.metadata["post_time"], 
                    "%Y-%m-%dT%H:%M:%S%z"
                ))
            except (ValueError, TypeError) as e:
                logger.warning(f"Unable to sort documents by timestamp: {str(e)}")

            return documents

        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}")
            return []

    def _format_metadata(self, metadata: Dict) -> Dict:
        """Format and standardize metadata"""
        return {
            "author": metadata.get("author", "Unknown"),
            "post_time": metadata.get("post_time", "Unknown"),
            "text": metadata.get("text", ""),
            "thread_title": metadata.get("thread_title", "Unknown Thread"),
            "thread_id": metadata.get("thread_id", "unknown"),
            "url": metadata.get("url", ""),
            "post_id": metadata.get("post_id", ""),
            "keywords": metadata.get("keywords", []),
            "sentiment": metadata.get("sentiment", 0),
            "total_posts": metadata.get("total_posts", 0),
            "declared_posts": metadata.get("declared_posts", 0)
        }