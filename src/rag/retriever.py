# retriever.py
import streamlit as st
from typing import List, Dict, Any
from langchain_core.documents import Document
import logging
from datetime import datetime
from config import EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class SmartRetriever:
    def __init__(self, index, embeddings):
        self.index = index
        self.embeddings = embeddings
        self.MAX_DOCUMENTS = 10000
        print(f"DEBUG: SmartRetriever initialized with EMBEDDING_DIMENSION: {EMBEDDING_DIMENSION}")

    def get_all_documents(self) -> List[Any]:
        """Recupera tutti i documenti dall'indice."""
        try:
            # Create a vector with correct dimension
            query_vector = [0.0] * EMBEDDING_DIMENSION
            query_vector[0] = 1.0
            
            print(f"DEBUG: get_all_documents - query vector dimension: {len(query_vector)}")
            st.write(f"DEBUG: query vector dimension before Pinecone call: {len(query_vector)}")
            
            results = self.index.query(
                vector=query_vector,
                top_k=self.MAX_DOCUMENTS,
                include_metadata=True
            )
            return results.matches
        except Exception as e:
            print(f"ERROR fetching documents: {str(e)}")
            st.error(f"Full error: {str(e)}")
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
            print(f"DEBUG: get_relevant_documents - query embedding dimension: {len(query_embedding)}")
            st.write(f"DEBUG: query embedding dimension before Pinecone call: {len(query_embedding)}")
            
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
                print("WARNING: Unable to sort documents by timestamp")
            
            return documents
            
        except Exception as e:
            print(f"ERROR in retrieval: {str(e)}")
            st.error(f"Full error in retrieval: {str(e)}")
            return [Document(
                page_content=f"Errore durante il recupero dei documenti: {str(e)}",
                metadata={"type": "error"}
            )]