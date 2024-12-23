# retriever.py
import streamlit as st
from typing import List, Dict, Any
from langchain_core.documents import Document
import logging
from datetime import datetime
from config import EMBEDDING_DIMENSION

class SmartRetriever:
    def __init__(self, index, embeddings):
        self.index = index
        self.embeddings = embeddings
        self.MAX_DOCUMENTS = 10000
        # Debug info direttamente in Streamlit
        st.write("🔍 Initializing Retriever:")
        st.write(f"- Expected dimension: {EMBEDDING_DIMENSION}")
        st.write(f"- Embeddings type: {type(self.embeddings)}")

    def get_all_documents(self) -> List[Any]:
        """Recupera tutti i documenti dall'indice."""
        try:
            # Crea vettore di test
            query_vector = [0.0] * EMBEDDING_DIMENSION
            query_vector[0] = 1.0
            
            # Debug info
            st.write("📊 Query vector info:")
            st.write(f"- Dimension: {len(query_vector)}")
            
            results = self.index.query(
                vector=query_vector,
                top_k=self.MAX_DOCUMENTS,
                include_metadata=True
            )
            return results.matches
        except Exception as e:
            st.error(f"Error in get_all_documents: {str(e)}")
            st.error(f"Query vector dimension was: {len(query_vector)}")
            return []

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Recupera documenti rilevanti usando Sentence Transformers."""
        try:
            # Genera l'embedding della query
            query_embedding = self.embeddings.embed_query(query)
            
            # Debug info
            st.write("🔍 Query embedding info:")
            st.write(f"- Dimension: {len(query_embedding)}")
            st.write(f"- Type: {type(query_embedding)}")
            
            results = self.index.query(
                vector=query_embedding,
                top_k=10,
                include_metadata=True
            )
            
            if not results.matches:
                return [Document(page_content="Nessun documento trovato", metadata={"type": "error"})]
            
            return self._process_results(results.matches)
            
        except Exception as e:
            st.error(f"Error in get_relevant_documents: {str(e)}")
            if 'query_embedding' in locals():
                st.error(f"Query embedding dimension was: {len(query_embedding)}")
            return [Document(
                page_content=f"Errore: {str(e)}",
                metadata={"type": "error"}
            )]

    def _process_results(self, matches) -> List[Document]:
        """Processa i risultati della query."""
        documents = []
        for match in matches:
            metadata = self._format_metadata(match.metadata)
            doc = Document(
                page_content=metadata["text"],
                metadata=metadata
            )
            documents.append(doc)
        
        try:
            documents.sort(key=lambda x: datetime.strptime(
                x.metadata["post_time"], 
                "%Y-%m-%dT%H:%M:%S%z"
            ))
        except:
            st.warning("Unable to sort documents by timestamp")
        
        return documents

    def _format_metadata(self, metadata: Dict) -> Dict:
        """Formatta i metadati."""
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