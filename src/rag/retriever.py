import streamlit as st
from typing import List
from langchain_core.documents import Document
from collections import defaultdict

class PineconeRetriever:
    def __init__(self, index, embeddings, top_k=10):  # Aumentato top_k a 10
        self.index = index
        self.embeddings = embeddings
        self.top_k = top_k
    
    def get_total_documents(self):
        try:
            stats = self.index.describe_index_stats()
            return stats.total_vector_count
        except Exception as e:
            st.error(f"Errore nel recupero statistiche: {str(e)}")
            return 0
    
    def get_threads_summary(self):
        """Create a summary of all threads in the database."""
        try:
            # Query with a zero vector to get all documents
            results = self.index.query(
                vector=[0] * 1536,  # dimensione standard per OpenAI embeddings
                top_k=1000,  # numero alto per prendere tutti i documenti
                include_metadata=True
            )
            
            # Raggruppa per thread_id
            threads = defaultdict(dict)
            for match in results.matches:
                thread_id = match.metadata.get('thread_id')
                if thread_id:
                    threads[thread_id].update({
                        'title': match.metadata.get('thread_title', 'Titolo non disponibile'),
                        'url': match.metadata.get('url', 'URL non disponibile')
                    })
            
            # Crea il sommario
            summary = "Riepilogo dei thread nel database:\n\n"
            for thread_info in threads.values():
                summary += f"- {thread_info['title']}\n"
            
            return summary
            
        except Exception as e:
            st.error(f"Errore nel recupero sommario: {str(e)}")
            return "Non è stato possibile recuperare il sommario dei thread."
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        try:
            # Add database stats and thread summary for overview queries
            stats_doc = Document(
                page_content=f"Il database contiene {self.get_total_documents()} posts totali.\n\n{self.get_threads_summary()}",
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