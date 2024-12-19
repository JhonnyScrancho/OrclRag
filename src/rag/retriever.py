import streamlit as st
from typing import List
from langchain_core.documents import Document

class PineconeRetriever:
    def __init__(self, index, embeddings, top_k=3):
        self.index = index
        self.embeddings = embeddings
        self.top_k = top_k
    
    def get_total_documents(self):
        """Get total number of documents in the index."""
        try:
            stats = self.index.describe_index_stats()
            return stats.total_vector_count
        except Exception as e:
            st.error(f"Errore nel recupero statistiche: {str(e)}")
            return 0
    
    def get_stats_metadata(self):
        """Create metadata about index statistics."""
        total_docs = self.get_total_documents()
        return f"Il database contiene attualmente {total_docs} posts totali."
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents for a query."""
        try:
            # Add stats to every query response
            stats_doc = Document(
                page_content=self.get_stats_metadata(),
                metadata={"type": "stats"}
            )
            
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Query the index
            results = self.index.query(
                vector=query_embedding,
                top_k=self.top_k,
                include_metadata=True
            )
            
            # Create regular documents
            documents = [stats_doc]  # Add stats as first document
            for result in results.matches:
                if result.metadata.get("text"):
                    documents.append(
                        Document(
                            page_content=result.metadata.get("text", ""),
                            metadata=result.metadata
                        )
                    )
            
            return documents
            
        except Exception as e:
            st.error(f"Errore nella ricerca documenti: {str(e)}")
            return []