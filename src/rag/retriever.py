# retriever.py
import streamlit as st
from typing import List, Dict, Any
from langchain_core.documents import Document
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SmartRetriever:
    def __init__(self, index, embeddings):
        self.index = index
        self.embeddings = embeddings
        self.MAX_DOCUMENTS = 10000

    def get_all_documents(self) -> List[Any]:
        """Recupera tutti i documenti dall'indice."""
        try:
            # Use embedding model to generate a query vector
            dummy_text = "query"
            query_vector = self.embeddings.embed_query(dummy_text)
            
            results = self.index.query(
                vector=query_vector,
                top_k=self.MAX_DOCUMENTS,
                include_metadata=True
            )
            return results.matches
        except Exception as e:
            logger.error(f"Error fetching documents: {str(e)}")
            return []

    def _format_metadata(self, metadata: Dict) -> Dict:
        """Formatta e standardizza i metadati."""
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
        """Recupera documenti rilevanti usando Sentence Transformers."""
        try:
            # Genera l'embedding della query
            query_embedding = self.embeddings.embed_query(query)
            
            # Cerca i documenti più simili
            results = self.index.query(
                vector=query_embedding,
                top_k=10,
                include_metadata=True
            )
            
            if not results.matches:
                return [Document(page_content="Nessun documento trovato", metadata={"type": "error"})]
            
            # Converti i match in Documents
            documents = []
            for match in results.matches:
                metadata = self._format_metadata(match.metadata)
                doc = Document(
                    page_content=metadata["text"],
                    metadata=metadata
                )
                documents.append(doc)
            
            # Ordina per timestamp
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
                page_content=f"Errore durante il recupero dei documenti: {str(e)}",
                metadata={"type": "error"}
            )]