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
        """Initialize the SmartRetriever with index and embeddings model."""
        self.index = index
        self.embeddings = embeddings
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
                vector=[1.0] + [0.0] * (self.embeddings.dimension - 1),
                filter={"thread_id": thread_id},
                top_k=1000,  # Increased to ensure we get all documents
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
            top_k = len(documents)  # Default to keeping all documents

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
        """Retrieve all documents from the index."""
        try:
            query_vector = [0.0] * self.embeddings.dimension
            query_vector[0] = 1.0
            
            # Use Streamlit's cache_data at function call level
            @st.cache_data(ttl=3600)
            def get_cached_docs(vector_key: tuple):
                return self._cached_documents(vector_key, 1000)  # Increased limit
            
            return get_cached_docs(tuple(query_vector))
            
        except Exception as e:
            logger.error(f"Error fetching all documents: {str(e)}")
            return []

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Enhanced document retrieval with complete thread fetching."""
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Prima query per trovare i thread rilevanti
            initial_results = self.index.query(
                vector=query_embedding,
                top_k=100,  # Aumentato per trovare più thread rilevanti
                include_metadata=True
            )
            
            if not initial_results.matches:
                logger.warning("No initial matches found")
                return []
            
            # Log per debug
            logger.info(f"Found {len(initial_results.matches)} initial matches")
            
            # Raccoglie thread_id unici
            thread_ids = {doc.metadata.get("thread_id") for doc in initial_results.matches if doc.metadata and doc.metadata.get("thread_id")}
            logger.info(f"Found {len(thread_ids)} unique threads")
            
            # Recupera TUTTI i post per ogni thread
            all_documents = []
            for thread_id in thread_ids:
                if not thread_id:
                    continue
                
                # Query specifica per il thread
                thread_results = self.index.query(
                    vector=[0.0] * self.embeddings.dimension,  # Vector neutro
                    filter={
                        "thread_id": thread_id,
                        "is_post": True  # Assicuriamoci di prendere solo i post
                    },
                    top_k=1000,  # Alto per prendere tutti i post
                    include_metadata=True
                )
                
                if thread_results and thread_results.matches:
                    thread_posts = thread_results.matches
                    logger.info(f"Retrieved {len(thread_posts)} posts from thread {thread_id}")
                    all_documents.extend(thread_posts)
            
            # Deduplicazione e conversione in Document
            seen_post_ids = set()
            documents = []
            
            for doc in all_documents:
                post_id = doc.metadata.get("unique_post_id")  # Usiamo unique_post_id invece di post_id
                if not post_id or post_id in seen_post_ids:
                    continue
                
                seen_post_ids.add(post_id)
                
                # Crea Document con tutti i metadati
                documents.append(Document(
                    page_content=doc.metadata.get("text", ""),
                    metadata={
                        "thread_id": doc.metadata.get("thread_id"),
                        "thread_title": doc.metadata.get("thread_title"),
                        "unique_post_id": post_id,
                        "post_id": doc.metadata.get("post_id"),
                        "author": doc.metadata.get("author"),
                        "post_time": doc.metadata.get("post_time"),
                        "total_posts": doc.metadata.get("total_posts"),
                        "declared_posts": doc.metadata.get("declared_posts"),
                        "sentiment": doc.metadata.get("sentiment", 0),
                        "keywords": doc.metadata.get("keywords", []),
                        "url": doc.metadata.get("url"),
                        "content_length": doc.metadata.get("content_length"),
                        "processing_time": doc.metadata.get("processing_time")
                    }
                ))
            
            # Ordina cronologicamente
            documents.sort(key=lambda x: x.metadata.get("post_time", ""))
            
            # Log statistiche finali
            logger.info(f"Retrieved and processed {len(documents)} total documents from {len(thread_ids)} threads")
            
            # Log completezza thread
            for thread_id in thread_ids:
                thread_posts = [d for d in documents if d.metadata.get("thread_id") == thread_id]
                if thread_posts:
                    declared = thread_posts[0].metadata.get("declared_posts", 0)
                    found = len(thread_posts)
                    logger.info(f"Thread {thread_id}: found {found}/{declared} posts")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}")
            return []

    def query_with_limit(self, query: str, limit: int = 5) -> List[Document]:
        """Query documents with a specific limit."""
        docs = self.get_relevant_documents(query)
        reranked_docs = self._rerank_documents(query, docs, top_k=limit)
        return reranked_docs