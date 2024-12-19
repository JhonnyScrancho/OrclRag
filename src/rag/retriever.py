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
            if not results.matches:
                st.warning("Nessun documento trovato nel database")
                return []
                
            st.write("Debug - Primo documento:", results.matches[0].metadata)
            return results.matches
        except Exception as e:
            st.error(f"Error fetching documents: {str(e)}")
            return []

    def extract_quotes_from_content(self, content: str) -> List[Dict]:
        """Estrae le citazioni dal contenuto di un post."""
        quotes = []
        lines = content.split('\n')
        current_quote = None

        for line in lines:
            if ' said:' in line:
                current_quote = {
                    'quoted_author': line.split(' said:')[0].strip(),
                    'quoted_content': ''
                }
            elif current_quote and 'Click to expand...' in line:
                if current_quote['quoted_content'].strip():
                    quotes.append(current_quote)
                current_quote = None
            elif current_quote:
                current_quote['quoted_content'] += line + '\n'

        return quotes

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents based on query type."""
        try:
            matches = self.get_all_documents()
            if not matches:
                return [Document(page_content="Database vuoto o errore di accesso.", metadata={"type": "error"})]

            # Organizziamo i thread e i post
            threads = defaultdict(lambda: {"posts": [], "title": "", "url": "", "scrape_time": ""})
            total_quotes = 0

            for match in matches:
                metadata = match.metadata
                text = metadata.get('text', '')
                thread_id = metadata.get('thread_id', 'unknown')
                
                # Aggiorniamo le informazioni del thread
                if not threads[thread_id]["title"]:
                    threads[thread_id].update({
                        "title": metadata.get('thread_title', ''),
                        "url": metadata.get('url', ''),
                        "scrape_time": metadata.get('scrape_time', '')
                    })

                # Analizziamo il contenuto per le citazioni
                quotes = self.extract_quotes_from_content(text)
                total_quotes += len(quotes)

                # Aggiungiamo il post alla lista del thread
                threads[thread_id]["posts"].append({
                    "author": metadata.get('author', ''),
                    "content": text,
                    "post_time": metadata.get('post_time', ''),
                    "quotes": quotes
                })

            # Gestiamo diversi tipi di query
            if any(keyword in query.lower() for keyword in ['quanti', 'numero', 'thread', 'post']):
                stats = f"""Nel database ho trovato:
- {len(threads)} thread
- {sum(len(t['posts']) for t in threads.values())} post totali"""
                return [Document(page_content=stats, metadata={"type": "stats"})]

            elif any(keyword in query.lower() for keyword in ['citazioni', 'quote', 'citano']):
                quotes_text = f"Nel database sono presenti {total_quotes} citazioni nei post."
                return [Document(page_content=quotes_text, metadata={"type": "quotes"})]

            elif any(keyword in query.lower() for keyword in ['riassunt', 'parlano', 'discute', 'riguarda']):
                summaries = []
                for thread_id, thread_data in threads.items():
                    if thread_data["posts"]:
                        first_post = next((p for p in thread_data["posts"] if not p["quotes"]), thread_data["posts"][0])
                        
                        summary = f"""Thread: "{thread_data['title']}"
URL: {thread_data['url']}

Post iniziale:
Autore: {first_post['author']}
Data: {first_post['post_time']}
Contenuto: {first_post['content']}

Statistiche discussione:
- Totale post: {len(thread_data['posts'])}
- Citazioni: {sum(len(p['quotes']) for p in thread_data['posts'])}"""
                        
                        summaries.append(Document(page_content=summary, metadata={"type": "summary"}))
                
                return summaries

            # Per altre query, utilizziamo la ricerca vettoriale
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