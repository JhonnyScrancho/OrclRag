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
        st.write(f"SmartRetriever initialized with embedding model type: {type(embeddings)}")

    def get_all_documents(self) -> List[Any]:
        """Recupera tutti i documenti dall'indice."""
        try:
            # Debug pre-query
            st.write("Creating query vector...")
            query_vector = [0.0] * 768
            query_vector[0] = 1.0
            
            st.write(f"Query vector type: {type(query_vector)}")
            st.write(f"Query vector dimension: {len(query_vector)}")
            st.write("First 5 elements:", query_vector[:5])
            
            # DEBUG: Stampa l'intero stack di chiamate
            import traceback
            st.write("Call stack:")
            st.text(''.join(traceback.format_stack()))
            
            # Stampa tutti gli oggetti che gestiranno il vettore
            st.write("Index type:", type(self.index))
            st.write("Index methods:", [m for m in dir(self.index) if not m.startswith('_')])
            
            # Prova la query con debug
            st.write("About to query with vector...")
            results = self.index.query(
                vector=query_vector,  # DEVE essere 768!
                top_k=self.MAX_DOCUMENTS,
                include_metadata=True
            )
            return results.matches
            
        except Exception as e:
            st.error("ERRORE DETTAGLIATO:")
            st.error(f"Type: {type(e)}")
            st.error(f"Message: {str(e)}")
            st.error(f"Query vector dim quando fallisce: {len(query_vector)}")
            import traceback
            st.error("Traceback completo:")
            st.error(traceback.format_exc())
            return []

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Recupera documenti rilevanti usando Sentence Transformers."""
        try:
            # Debug pre-embedding
            st.write("About to generate embedding for query:", query)
            query_embedding = self.embeddings.embed_query(query)
            st.write(f"Generated embedding dimension: {len(query_embedding)}")
            st.write(f"Embedding type: {type(query_embedding)}")
            
            results = self.index.query(
                vector=query_embedding,
                top_k=10,
                include_metadata=True
            )
            
            if not results.matches:
                return [Document(page_content="Nessun documento trovato", metadata={"type": "error"})]
            
            documents = []
            for match in results.matches:
                metadata = self._format_metadata(match.metadata)
                doc = Document(
                    page_content=metadata["text"],
                    metadata=metadata
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            st.error(f"Error in retrieval: {str(e)}")
            st.error(f"Query embedding dimension when failed: {len(query_embedding)}")
            return [Document(
                page_content=f"Errore durante il recupero dei documenti: {str(e)}",
                metadata={"type": "error"}
            )]

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