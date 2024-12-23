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
        st.write("Init SmartRetriever")

    def get_all_documents(self) -> List[Any]:
        """Recupera tutti i documenti dall'indice."""
        try:
            # Crea il vettore
            query_vector = [0.0] * 768
            query_vector[0] = 1.0
            
            # Ispeziona il vettore
            st.write("Vector inspection before Pinecone query:")
            st.write(f"- Type: {type(query_vector)}")
            st.write(f"- Length: {len(query_vector)}")
            st.write(f"- First 5 elements: {query_vector[:5]}")
            st.write(f"- Last 5 elements: {query_vector[-5:]}")
            
            # Ispeziona l'oggetto index
            st.write("\nPinecone index inspection:")
            st.write(f"- Type: {type(self.index)}")
            st.write(f"- Dir: {dir(self.index)}")
            
            # Debug della chiamata
            st.write("\nPreparing query with:")
            query_args = {
                "vector": query_vector,
                "top_k": self.MAX_DOCUMENTS,
                "include_metadata": True
            }
            st.write(query_args)
            
            results = self.index.query(**query_args)
            return results.matches
        except Exception as e:
            st.error("Error in get_all_documents:")
            st.error(f"- Error type: {type(e)}")
            st.error(f"- Error message: {str(e)}")
            st.error(f"- Dir of error: {dir(e)}")
            return []

    def _format_metadata(self, metadata: Dict) -> Dict:
        """Formatta e standardizza i metadati."""
        return {
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

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Recupera documenti rilevanti usando Sentence Transformers."""
        try:
            # Genera l'embedding della query
            query_embedding = self.embeddings.embed_query(query)
            
            st.write(f"Query embedding dimension: {len(query_embedding)}")
            
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
            st.error(f"Error in retrieval: {str(e)}")
            return [Document(
                page_content=f"Errore durante il recupero dei documenti: {str(e)}",
                metadata={"type": "error"}
            )]