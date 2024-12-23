# retriever.py
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
        self.EMBEDDING_DIMENSION = 768  # Set explicitly to match index dimension

    def get_all_documents(self) -> List[Any]:
        """Retrieve all documents from the index using a properly dimensioned query vector."""
        try:
            # Create a zero vector with correct dimension
            query_vector = [0.0] * self.EMBEDDING_DIMENSION
            # Set first element to 1.0 to ensure non-zero vector
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
        """Retrieve relevant documents using proper embeddings."""
        try:
            # Generate query embedding using the embeddings model
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
            
            # Convert matches to Documents
            documents = []
            for match in results.matches:
                metadata = self._format_metadata(match.metadata)
                doc = Document(
                    page_content=metadata["text"],
                    metadata=metadata
                )
                documents.append(doc)
            
            # Sort by timestamp
            try:
                documents.sort(key=lambda x: datetime.strptime(
                    x.metadata["post_time"], 
                    "%Y-%m-%dT%H:%M:%S%z"
                ))
            except (ValueError, TypeError):
                logger.warning("Unable to sort documents by timestamp")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}")
            return [Document(
                page_content=f"Error retrieving documents: {str(e)}",
                metadata={"type": "error"}
            )]