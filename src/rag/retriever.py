import streamlit as st
from typing import List, Dict, Any
from langchain_core.documents import Document
import logging
from datetime import datetime
from config import EMBEDDING_DIMENSION

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
            
            # Raggruppa i chunks per thread_id
            grouped_chunks = {}
            for match in results.matches:
                thread_id = match.metadata.get("thread_id", "unknown")
                if thread_id not in grouped_chunks:
                    grouped_chunks[thread_id] = []
                grouped_chunks[thread_id].append(match)
            
            # Ricostruisci i documenti completi
            complete_documents = []
            for thread_chunks in grouped_chunks.values():
                # Ordina i chunks per timestamp del post
                thread_chunks.sort(key=lambda x: datetime.fromisoformat(
                    x.metadata.get("post_time", "1970-01-01T00:00:00+00:00")
                ))
                
                # Ricostruisci ogni post del thread
                for chunk in thread_chunks:
                    # Usa il testo originale se disponibile
                    if "text" in chunk.metadata:
                        metadata = chunk.metadata.copy()
                        # Aggiungi informazioni del thread
                        metadata.update({
                            "thread_title": chunk.metadata.get("thread_title", "Unknown Thread"),
                            "url": chunk.metadata.get("url", ""),
                            "scrape_time": chunk.metadata.get("scrape_time", "")
                        })
                        
                        complete_documents.append(Document(
                            page_content=chunk.metadata["text"],
                            metadata=metadata
                        ))
            
            logger.info(f"Retrieved and reconstructed {len(complete_documents)} documents")
            return complete_documents
            
        except Exception as e:
            logger.error(f"Error fetching documents: {str(e)}")
            return []

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
                top_k=self.MAX_DOCUMENTS,
                include_metadata=True
            )
            
            if not results.matches:
                return [Document(page_content="No documents found", metadata={"type": "error"})]
            
            # Raggruppa i chunks per thread_id
            grouped_chunks = {}
            for match in results.matches:
                thread_id = match.metadata.get("thread_id", "unknown")
                if thread_id not in grouped_chunks:
                    grouped_chunks[thread_id] = []
                grouped_chunks[thread_id].append(match)
            
            # Ricostruisci i documenti completi
            relevant_documents = []
            for thread_chunks in grouped_chunks.values():
                # Ordina i chunks per timestamp del post
                thread_chunks.sort(key=lambda x: datetime.fromisoformat(
                    x.metadata.get("post_time", "1970-01-01T00:00:00+00:00")
                ))
                
                # Ricostruisci ogni post del thread
                for chunk in thread_chunks:
                    if "text" in chunk.metadata:
                        metadata = chunk.metadata.copy()
                        # Aggiungi informazioni del thread
                        metadata.update({
                            "thread_title": chunk.metadata.get("thread_title", "Unknown Thread"),
                            "url": chunk.metadata.get("url", ""),
                            "scrape_time": chunk.metadata.get("scrape_time", "")
                        })
                        
                        relevant_documents.append(Document(
                            page_content=chunk.metadata["text"],
                            metadata=metadata
                        ))
            
            return relevant_documents
            
        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}")
            return [Document(
                page_content=f"Error retrieving documents: {str(e)}",
                metadata={"type": "error"}
            )]