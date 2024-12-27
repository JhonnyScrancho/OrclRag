import streamlit as st
from typing import List, Dict, Any
from langchain_core.documents import Document
import logging
from datetime import datetime
from config import EMBEDDING_DIMENSION
import json
from operator import itemgetter

logger = logging.getLogger(__name__)

class SmartRetriever:
    def __init__(self, index, embeddings):
        self.index = index
        self.embeddings = embeddings
        self.MAX_DOCUMENTS = 10000
        self.EMBEDDING_DIMENSION = EMBEDDING_DIMENSION

    def get_all_documents(self) -> List[Document]:
        """Retrieve and reconstruct all documents from the index."""
        try:
            # Create a query that will match all documents
            query_vector = [0.0] * self.EMBEDDING_DIMENSION
            query_vector[0] = 1.0  # Set first element to 1.0 to ensure non-zero vector
            
            logger.info("Retrieving all documents from index...")
            
            results = self.index.query(
                vector=query_vector,
                top_k=self.MAX_DOCUMENTS,
                include_metadata=True
            )
            
            if not results.matches:
                logger.warning("No documents found in index")
                return []
            
            # Raggruppa i chunks per post_id
            grouped_chunks = {}
            for match in results.matches:
                post_id = match.metadata.get("post_id", "unknown")
                if post_id not in grouped_chunks:
                    grouped_chunks[post_id] = []
                grouped_chunks[post_id].append(match)
            
            # Ricostruisci i documenti completi
            complete_documents = []
            for post_chunks in grouped_chunks.values():
                # Ordina i chunks per numero
                post_chunks.sort(key=lambda x: x.metadata.get("chunk_number", 0))
                
                # Usa il testo originale completo se disponibile, altrimenti ricostruisci dai chunks
                if "text" in post_chunks[0].metadata:
                    complete_text = post_chunks[0].metadata["text"]
                else:
                    complete_text = " ".join(chunk.metadata.get("chunk_text", "") for chunk in post_chunks)
                
                # Prendi i metadati dal primo chunk e rimuovi i campi specifici del chunking
                metadata = post_chunks[0].metadata.copy()
                metadata.pop("chunk_number", None)
                metadata.pop("total_chunks", None)
                metadata.pop("chunk_text", None)
                
                complete_documents.append(Document(
                    page_content=complete_text,
                    metadata=metadata
                ))
            
            # Ordina per timestamp se possibile
            try:
                complete_documents.sort(key=lambda x: datetime.fromisoformat(x.metadata.get("post_time", "1970-01-01T00:00:00+00:00")))
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not sort documents by timestamp: {e}")
            
            logger.info(f"Retrieved and reconstructed {len(complete_documents)} documents")
            return complete_documents
            
        except Exception as e:
            logger.error(f"Error fetching documents: {str(e)}")
            return []

    def _format_metadata(self, metadata: Dict) -> Dict:
        """Format and standardize metadata."""
        formatted = {
            "author": metadata.get("author", "Unknown"),
            "post_time": metadata.get("post_time", "Unknown"),
            "text": metadata.get("text", ""),
            "thread_title": metadata.get("thread_title", "Unknown Thread"),
            "thread_id": metadata.get("thread_id", "unknown"),
            "url": metadata.get("url", ""),
            "post_id": metadata.get("post_id", ""),
            "keywords": metadata.get("keywords", []),
            "sentiment": metadata.get("sentiment", 0)
        }
        return formatted

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents using semantic search."""
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Verify embedding dimension
            if len(query_embedding) != self.EMBEDDING_DIMENSION:
                raise ValueError(f"Query embedding dimension {len(query_embedding)} does not match index dimension {self.EMBEDDING_DIMENSION}")
            
            # Search for similar documents
            results = self.index.query(
                vector=query_embedding,
                top_k=10,
                include_metadata=True
            )
            
            if not results.matches:
                return [Document(page_content="No documents found", metadata={"type": "error"})]
            
            # Raggruppa i chunks per post_id
            grouped_chunks = {}
            for match in results.matches:
                post_id = match.metadata.get("post_id", "unknown")
                if post_id not in grouped_chunks:
                    grouped_chunks[post_id] = []
                grouped_chunks[post_id].append(match)
            
            # Ricostruisci i documenti completi
            relevant_documents = []
            for post_chunks in grouped_chunks.values():
                # Ordina i chunks per numero
                post_chunks.sort(key=lambda x: x.metadata.get("chunk_number", 0))
                
                # Usa il testo originale se disponibile
                if "text" in post_chunks[0].metadata:
                    complete_text = post_chunks[0].metadata["text"]
                else:
                    complete_text = " ".join(chunk.metadata.get("chunk_text", "") for chunk in post_chunks)
                
                # Prendi i metadati dal primo chunk e rimuovi i campi del chunking
                metadata = self._format_metadata(post_chunks[0].metadata)
                
                relevant_documents.append(Document(
                    page_content=complete_text,
                    metadata=metadata
                ))
            
            # Ordina per timestamp
            try:
                relevant_documents.sort(key=lambda x: datetime.fromisoformat(x.metadata["post_time"]))
            except (ValueError, TypeError):
                logger.warning("Unable to sort documents by timestamp")
            
            return relevant_documents
            
        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}")
            return [Document(
                page_content=f"Error retrieving documents: {str(e)}",
                metadata={"type": "error"}
            )]