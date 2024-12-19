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

    def parse_content(self, text: str) -> Dict:
        """Estrae le informazioni strutturate dal testo."""
        lines = text.split('\n')
        data = {}
        current_field = None
        
        for line in lines:
            if ': ' in line:
                key, value = line.split(': ', 1)
                data[key.strip()] = value.strip()
                
        return data

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
                
                # Estrai citazioni
                quotes = self.extract_quotes(content)
                quotes_count += len(quotes)
                
                # Aggiungi al thread appropriato
                threads[thread_id].append({
                    'metadata': metadata,
                    'content': content,
                    'quotes': quotes,
                    'parsed_content': self.parse_content(content)
                })

            # Query per statistiche
            if any(keyword in query.lower() for keyword in ['quanti', 'numero', 'thread', 'post']):
                stats = f"Nel database ho trovato {len(threads)} thread e {len(matches)} post."
                return [Document(page_content=stats, metadata={"type": "stats"})]

            # Query per citazioni
            elif any(keyword in query.lower() for keyword in ['citazioni', 'quote', 'citano']):
                quote_stats = f"Nel database sono presenti {quotes_count} citazioni."
                return [Document(page_content=quote_stats, metadata={"type": "quotes"})]

            # Query per riassunto thread
            elif any(keyword in query.lower() for keyword in ['riassunt', 'parlano', 'discute', 'riguarda']):
                summaries = []
                for thread_id, posts in threads.items():
                    if not posts:
                        continue
                        
                    # Prendi i metadati del thread dal primo post
                    thread_meta = posts[0]['metadata']
                    
                    # Trova il primo post (quello senza citazioni)
                    first_post = next((post for post in posts if not post['quotes']), posts[0])
                    
                    summary = f"""Titolo thread: {thread_meta.get('thread_title')}
URL: {thread_meta.get('url')}

Post iniziale:
Autore: {first_post['parsed_content'].get('Author', 'Sconosciuto')}
Contenuto: {first_post['parsed_content'].get('Content', 'Non disponibile')}

La discussione contiene {len(posts)} post totali."""

                    summaries.append(Document(page_content=summary, metadata={"type": "summary"}))
                
                return summaries if summaries else [Document(page_content="Nessun thread trovato.", metadata={"type": "summary"})]

            # Per altre query, usa la ricerca vettoriale
            query_embedding = self.embeddings.embed_query(query)
            results = self.index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True
            )
            
            return [
                Document(page_content=match.metadata.get("text", ""), metadata=match.metadata)
                for match in results.matches
            ]

        except Exception as e:
            st.error(f"Error in retrieval: {str(e)}")
            return [Document(page_content=f"Errore durante il recupero dei documenti: {str(e)}", metadata={"type": "error"})]