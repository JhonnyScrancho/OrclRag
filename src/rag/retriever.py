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

    def parse_post_content(self, text: str) -> Dict:
        """Estrae le informazioni strutturate dal testo del post."""
        content_dict = {}
        for line in text.split('\n'):
            if ': ' in line:
                key, value = line.split(': ', 1)
                content_dict[key.strip()] = value.strip()
        return content_dict

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
                
                # Parse del contenuto strutturato
                parsed_content = self.parse_post_content(content)
                
                # Aggiungi al thread con il contenuto parsato
                threads[thread_id].append({
                    'metadata': metadata,
                    'parsed_content': parsed_content,
                    'quotes': self.extract_quotes(content)
                })
                quotes_count += len(self.extract_quotes(content))

            # Query per statistiche
            if any(keyword in query.lower() for keyword in ['quanti', 'numero', 'thread', 'post']):
                return [Document(page_content=f"Ho trovato {len(threads)} thread e {len(matches)} post nel database.", metadata={"type": "stats"})]

            # Query per contenuto/riassunto thread
            elif any(keyword in query.lower() for keyword in ['riassunt', 'parlano', 'discute', 'riguarda', 'cosa', 'contenuto']):
                summaries = []
                for thread_id, posts in threads.items():
                    # Trova il primo post senza citazioni
                    first_post = next((post for post in posts if not post['quotes']), posts[0])
                    
                    # Estrai i metadati del thread
                    thread_meta = first_post['metadata']
                    parsed_content = first_post['parsed_content']
                    
                    summary = f"""Titolo thread: {thread_meta.get('thread_title', 'N/A')}

Contenuto iniziale:
{parsed_content.get('Content', 'Non disponibile')}

Keywords principali: {parsed_content.get('Keywords', 'Non disponibili')}"""

                    summaries.append(Document(page_content=summary, metadata={"type": "summary"}))
                
                return summaries

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