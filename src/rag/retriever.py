import streamlit as st
from typing import List, Dict
from langchain_core.documents import Document
from collections import defaultdict

class PineconeRetriever:
    def __init__(self, index, embeddings):
        self.index = index
        self.embeddings = embeddings
    
    def get_all_documents(self):
        """Recupera tutti i documenti dall'indice."""
        try:
            results = self.index.query(
                vector=[0] * 1536,
                top_k=10000,
                include_metadata=True
            )
            return results.matches
        except Exception as e:
            st.error(f"Error fetching documents: {str(e)}")
            return []

    def extract_quotes(self, content: str) -> List[Dict]:
        """Estrae le citazioni dal contenuto."""
        quotes = []
        if ' said:' in content and 'Click to expand...' in content:
            parts = content.split(' said:', 1)
            if len(parts) == 2:
                author = parts[0].strip()
                quoted_content = parts[1].split('Click to expand...')[0].strip()
                quotes.append({
                    'author': author,
                    'content': quoted_content
                })
        return quotes

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents based on query type."""
        try:
            matches = self.get_all_documents()
            if not matches:
                return [Document(page_content="Database vuoto o errore di accesso.", metadata={"type": "error"})]

            # Raggruppa per thread_id
            threads = defaultdict(list)
            quotes_count = 0
            
            for match in matches:
                metadata = match.metadata
                thread_id = metadata.get('thread_id', 'unknown')
                content = metadata.get('text', '')
                quotes = self.extract_quotes(content)
                quotes_count += len(quotes)
                threads[thread_id].append(metadata)

            # Query per statistiche
            if any(keyword in query.lower() for keyword in ['quanti', 'numero', 'thread', 'post']):
                return [Document(page_content=f"Nel database ho trovato {len(threads)} thread e {len(matches)} post.", metadata={"type": "stats"})]

            # Query per citazioni
            elif any(keyword in query.lower() for keyword in ['citazioni', 'quote', 'citano']):
                return [Document(page_content=f"Nel database sono presenti {quotes_count} citazioni.", metadata={"type": "quotes"})]

            # Query per riassunto thread
            elif any(keyword in query.lower() for keyword in ['riassunt', 'parlano', 'discute', 'riguarda']):
                thread_summaries = []
                for thread_id, posts in threads.items():
                    if posts:
                        first_post = posts[0]
                        summary = f"""Titolo: {first_post.get('thread_title')}
URL: {first_post.get('url')}

Argomento: {first_post.get('content', '').split('Content: ')[1].split('Keywords:')[0].strip() if 'Content: ' in first_post.get('content', '') else 'Non disponibile'}

Totale post: {len(posts)}"""
                        thread_summaries.append(Document(page_content=summary, metadata={"type": "summary"}))
                return thread_summaries

            # Per altre query, usa la ricerca vettoriale
            query_embedding = self.embeddings.embed_query(query)
            results = self.index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True
            )
            
            return [Document(page_content=match.metadata.get("text", ""), metadata=match.metadata)
                    for match in results.matches]

        except Exception as e:
            st.error(f"Error in retrieval: {str(e)}")
            return [Document(page_content=f"Errore durante il recupero dei documenti: {str(e)}", metadata={"type": "error"})]