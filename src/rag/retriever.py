import streamlit as st
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from collections import defaultdict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class PineconeRetriever:
    def __init__(self, index, embeddings):
        self.index = index
        self.embeddings = embeddings
        self.SIMILARITY_THRESHOLD = 0.75
        self.MAX_DOCUMENTS = 10

    def get_all_documents(self) -> List[Any]:
        """Recupera tutti i documenti dall'indice."""
        try:
            results = self.index.query(
                vector=[0] * 1536,
                top_k=10000,
                include_metadata=True
            )
            return results.matches
        except Exception as e:
            logger.error(f"Error fetching documents: {str(e)}")
            return []

    def parse_post_content(self, text: str) -> Dict[str, str]:
        """Estrae informazioni strutturate dal testo del post."""
        content_dict = {}
        current_section = None
        current_content = []

        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue

            if ': ' in line:
                if current_section:
                    content_dict[current_section] = ' '.join(current_content)
                    current_content = []
                
                key, value = line.split(': ', 1)
                current_section = key.strip()
                current_content = [value.strip()]
            else:
                current_content.append(line)

        if current_section:
            content_dict[current_section] = ' '.join(current_content)

        return content_dict

    def extract_quotes(self, content: str) -> List[Dict[str, str]]:
        """Estrae le citazioni dal contenuto del post."""
        quotes = []
        segments = content.split('Click to expand...')
        
        for segment in segments[:-1]:  # Ignora l'ultimo segmento
            if ' said:' in segment:
                parts = segment.rsplit(' said:', 1)  # Usa rsplit per gestire nomi con "said"
                if len(parts) == 2:
                    author = parts[0].strip()
                    quoted_content = parts[1].strip()
                    quotes.append({
                        'author': author,
                        'content': quoted_content,
                        'timestamp': datetime.now().isoformat()
                    })
        return quotes

    def get_thread_content(self, matches: List[Any]) -> List[Document]:
        """Estrae e formatta il contenuto dei thread dai matches."""
        documents = []
        
        for match in matches:
            metadata = match.metadata
            content = metadata.get("text", "")
            parsed = self.parse_post_content(content)
            
            formatted_content = f"""
Thread: {metadata.get('thread_title', 'N/A')}
URL: {metadata.get('url', 'N/A')}
Author: {parsed.get('Author', 'Unknown')}
Time: {parsed.get('Time', 'Unknown')}
Content: {parsed.get('Content', 'No content available')}
Keywords: {parsed.get('Keywords', 'N/A')}
"""
            
            # Estrai citazioni se presenti
            quotes = self.extract_quotes(parsed.get('Content', ''))
            if quotes:
                formatted_content += "\nCitazioni:\n"
                for quote in quotes:
                    formatted_content += f"- {quote['author']} ha scritto: {quote['content']}\n"
            
            documents.append(Document(
                page_content=formatted_content,
                metadata={
                    **metadata,
                    "author": parsed.get('Author', 'Unknown'),
                    "post_time": parsed.get('Time', 'Unknown'),
                    "has_quotes": bool(quotes)
                }
            ))
        
        return documents

    def process_thread_documents(self, matches: List[Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Organizza i documenti per thread."""
        threads = defaultdict(list)
        
        for match in matches:
            metadata = match.metadata
            thread_id = metadata.get('thread_id', 'unknown')
            content = metadata.get('text', '')
            
            parsed_content = self.parse_post_content(content)
            quotes = self.extract_quotes(content)
            
            threads[thread_id].append({
                'metadata': metadata,
                'parsed_content': parsed_content,
                'quotes': quotes,
                'similarity_score': getattr(match, 'score', 0)
            })
            
        return threads

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Recupera i documenti rilevanti basati sulla query."""
        try:
            # Gestione query statistiche
            if any(keyword in query.lower() for keyword in ['quanti', 'numero', 'thread', 'post']):
                matches = self.get_all_documents()
                if not matches:
                    return [Document(
                        page_content="Database vuoto o inaccessibile.",
                        metadata={"type": "error"}
                    )]
                    
                threads = self.process_thread_documents(matches)
                stats = f"Database contiene {len(threads)} thread e {len(matches)} post totali."
                return [Document(page_content=stats, metadata={"type": "stats"})]

            # Gestione query di contenuto
            if any(keyword in query.lower() for keyword in ['parlano', 'discute', 'riguarda', 'cosa', 'contenuto']):
                matches = self.get_all_documents()
                if not matches:
                    return [Document(
                        page_content="Database vuoto o inaccessibile.",
                        metadata={"type": "error"}
                    )]
                
                return self.get_thread_content(matches)

            # Ricerca semantica per altre query
            query_embedding = self.embeddings.embed_query(query)
            results = self.index.query(
                vector=query_embedding,
                top_k=self.MAX_DOCUMENTS,
                include_metadata=True
            )
            
            relevant_docs = []
            for match in results.matches:
                if hasattr(match, 'score') and match.score >= self.SIMILARITY_THRESHOLD:
                    relevant_docs.extend(self.get_thread_content([match]))
            
            if not relevant_docs:
                return [Document(
                    page_content="Non ho trovato documenti sufficientemente rilevanti per la tua query.",
                    metadata={"type": "no_results"}
                )]
            
            return relevant_docs

        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}", exc_info=True)
            return [Document(
                page_content=f"Errore durante il recupero dei documenti: {str(e)}",
                metadata={"type": "error"}
            )]

    def get_thread_statistics(self) -> Dict[str, Any]:
        """Calcola statistiche dettagliate sui thread."""
        try:
            matches = self.get_all_documents()
            threads = self.process_thread_documents(matches)
            
            stats = {
                "total_threads": len(threads),
                "total_posts": len(matches),
                "threads_details": []
            }
            
            for thread_id, posts in threads.items():
                first_post = posts[0] if posts else {}
                thread_meta = first_post.get('metadata', {})
                
                # Calcola statistiche per il thread
                thread_stats = {
                    "thread_id": thread_id,
                    "title": thread_meta.get('thread_title', 'N/A'),
                    "url": thread_meta.get('url', 'N/A'),
                    "post_count": len(posts),
                    "first_post_date": thread_meta.get('scrape_time', 'N/A'),
                    "unique_authors": len(set(p['metadata'].get('author', '') for p in posts)),
                    "has_quotes": any(p.get('quotes') for p in posts),
                    "average_content_length": sum(len(p['parsed_content'].get('Content', '')) for p in posts) / len(posts) if posts else 0
                }
                stats["threads_details"].append(thread_stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}", exc_info=True)
            return {"error": str(e)}