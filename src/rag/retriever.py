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

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents based on query type."""
        try:
            matches = self.get_all_documents()
            
            # Organizza i dati del thread
            thread_data = defaultdict(list)
            quotes = []
            
            for match in matches:
                metadata = match.metadata
                content = metadata.get('text', '')
                thread_data[metadata.get('thread_id', '')].append({
                    'content': content,
                    'author': metadata.get('author', ''),
                    'post_time': metadata.get('post_time', ''),
                    'title': metadata.get('thread_title', '')
                })
                
                # Identifica le citazioni
                if ' said:' in content and 'Click to expand...' in content:
                    quotes.append(content)

            # Query per statistiche
            if any(keyword in query.lower() for keyword in ['quanti', 'numero', 'thread', 'post']):
                stats = f"""Nel database ho trovato:
- {len(thread_data)} thread di discussione
- {len(matches)} post totali"""
                return [Document(page_content=stats, metadata={"type": "stats"})]

            # Query per citazioni
            elif any(keyword in query.lower() for keyword in ['citazioni', 'quote', 'citano']):
                quote_text = f"""Ho identificato {len(quotes)} citazioni nei thread.
Queste citazioni rappresentano le interazioni tra gli utenti dove fanno riferimento a messaggi precedenti."""
                return [Document(page_content=quote_text, metadata={"type": "quotes"})]

            # Query per riassunto thread
            elif any(keyword in query.lower() for keyword in ['riassunt', 'parlano', 'discute', 'riguarda']):
                summaries = []
                for thread_id, posts in thread_data.items():
                    if posts:
                        thread_title = posts[0]['title']
                        # Trova il primo post (quello che inizia la discussione)
                        first_post = next((post for post in posts if ' said:' not in post['content']), None)
                        
                        if first_post:
                            summary = f"""Thread: "{thread_title}"

Contenuto iniziale della discussione:
{first_post['content']}

Dettagli della discussione:
- Numero totale di post: {len(posts)}
- Primo post da: {first_post['author']}
- Data inizio: {first_post['post_time']}"""
                            
                            summaries.append(Document(page_content=summary, metadata={"type": "summary"}))
                
                return summaries if summaries else [Document(page_content="Non ho trovato thread da riassumere nel database.", metadata={"type": "summary"})]

            # Per altre query, usa la ricerca vettoriale standard
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
            return []