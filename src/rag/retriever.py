from sentence_transformers import CrossEncoder
import streamlit as st
from typing import List, Dict, Any
from langchain_core.documents import Document
import logging
from datetime import datetime
from config import EMBEDDING_DIMENSION
from functools import lru_cache

logger = logging.getLogger(__name__)

class SmartRetriever:
    def __init__(self, index, embeddings):
        self.index = index
        self.embeddings = embeddings
        self.MAX_DOCUMENTS = 10000
        self.EMBEDDING_DIMENSION = EMBEDDING_DIMENSION
        # Initialize cross-encoder for re-ranking
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def _cached_documents(self, query_vector: List[float], top_k: int):
        """Internal cached function for document retrieval"""
        # Convert query_vector to tuple for hashing
        query_key = tuple(query_vector)
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )
            return results.matches if results and hasattr(results, 'matches') else []
        except Exception as e:
            logger.error(f"Error in cached document retrieval: {str(e)}")
            return []

    def get_all_documents(self) -> List[Any]:
        """Retrieve all documents using zero vector with first element as 1"""
        try:
            query_vector = [0.0] * self.EMBEDDING_DIMENSION
            query_vector[0] = 1.0
            
            # Use Streamlit's cache_data at function call level
            @st.cache_data(ttl=3600)
            def get_cached_docs(vector_key: tuple):
                return self._cached_documents(list(vector_key), self.MAX_DOCUMENTS)
            
            return get_cached_docs(tuple(query_vector))
            
        except Exception as e:
            logger.error(f"Error fetching all documents: {str(e)}")
            return []

    def _rerank_documents(self, query: str, documents: List[Document], top_k: int = 10) -> List[Document]:
        """Re-rank documents using cross-encoder"""
        if not documents:
            return []

        # Prepare pairs for cross-encoder
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)
        
        # Create document-score pairs and sort
        doc_score_pairs = list(zip(documents, scores))
        reranked_docs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        
        # Return top-k documents
        return [doc for doc, _ in reranked_docs[:top_k]]

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Enhanced document retrieval with re-ranking"""
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            if len(query_embedding) != self.EMBEDDING_DIMENSION:
                raise ValueError(f"Query embedding dimension mismatch")
            
            # Use cached retrieval for initial documents
            @st.cache_data(ttl=3600)
            def get_cached_relevant_docs(query_key: str, vector_key: tuple):
                docs = self._cached_documents(list(vector_key), 20)  # Get more for re-ranking
                return [
                    Document(
                        page_content=doc.metadata.get("text", ""),
                        metadata=self._format_metadata(doc.metadata)
                    ) for doc in docs
                ]
            
            # Get initial documents
            documents = get_cached_relevant_docs(query, tuple(query_embedding))
            
            if not documents:
                return [Document(page_content="No documents found", metadata={"type": "error"})]
            
            # Re-rank documents (not cached as it's query-dependent)
            reranked_docs = self._rerank_documents(query, documents)
            
            # Sort by timestamp
            try:
                reranked_docs.sort(key=lambda x: datetime.strptime(
                    x.metadata["post_time"], 
                    "%Y-%m-%dT%H:%M:%S%z"
                ))
            except (ValueError, TypeError):
                logger.warning("Unable to sort documents by timestamp")
            
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}")
            return [Document(
                page_content=f"Error retrieving documents: {str(e)}",
                metadata={"type": "error"}
            )]

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
            "chunk_index": metadata.get("chunk_index", 0),
            "total_chunks": metadata.get("total_chunks", 1)
        }