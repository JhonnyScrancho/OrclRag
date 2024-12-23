from venv import logger
from sentence_transformers import CrossEncoder
import streamlit as st
from typing import List, Dict, Any
from langchain_core.documents import Document
import logging
from datetime import datetime
from config import EMBEDDING_DIMENSION
from functools import lru_cache

class SmartRetriever:
    def __init__(self, index, embeddings):
        self.index = index
        self.embeddings = embeddings
        self.MAX_DOCUMENTS = 10000
        self.EMBEDDING_DIMENSION = EMBEDDING_DIMENSION
        # Inizializza il cross-encoder per re-ranking
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        # Cache TTL in secondi (1 ora)
        self.CACHE_TTL = 3600

    @st.cache_data(ttl=3600)
    def get_all_documents(self) -> List[Any]:
        """Cached version of document retrieval"""
        return self._get_all_documents_impl()

    def _get_all_documents_impl(self) -> List[Any]:
        """Actual implementation of document retrieval"""
        try:
            query_vector = [0.0] * EMBEDDING_DIMENSION
            query_vector[0] = 1.0
            
            results = self.index.query(
                vector=query_vector,
                top_k=self.MAX_DOCUMENTS,
                include_metadata=True
            )
            
            if not results.matches:
                logger.warning("No documents found in index")
                return []
                
            return results.matches
            
        except Exception as e:
            logger.error(f"Error fetching documents: {str(e)}")
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

    @st.cache_data(ttl=3600)
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Enhanced document retrieval with re-ranking and caching"""
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            if len(query_embedding) != self.EMBEDDING_DIMENSION:
                raise ValueError(f"Query embedding dimension mismatch")
            
            # Initial retrieval (get more documents for re-ranking)
            results = self.index.query(
                vector=query_embedding,
                top_k=20,  # Get more docs for re-ranking
                include_metadata=True
            )
            
            if not results.matches:
                return [Document(page_content="No documents found", metadata={"type": "error"})]
            
            # Convert matches to Documents
            documents = []
            for match in results.matches:
                metadata = self._format_metadata(match.metadata)
                doc = Document(
                    page_content=metadata["text"],
                    metadata=metadata
                )
                documents.append(doc)
            
            # Re-rank documents
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
        """Format and standardize metadata with additional fields"""
        formatted = {
            "author": metadata.get("author", "Unknown"),
            "post_time": metadata.get("post_time", "Unknown"),
            "text": metadata.get("text", ""),
            "thread_title": metadata.get("thread_title", "Unknown Thread"),
            "thread_id": metadata.get("thread_id", "unknown"),
            "url": metadata.get("url", ""),
            "post_id": metadata.get("post_id", ""),
            "keywords": metadata.get("keywords", []),
            "sentiment": metadata.get("sentiment", 0),
            "chunk_index": metadata.get("chunk_index", 0),  # Added for better context
            "total_chunks": metadata.get("total_chunks", 1)  # Added for better context
        }
        return formatted