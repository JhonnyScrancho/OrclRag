import re
import streamlit as st
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from collections import defaultdict
import logging
import json
import statistics
from datetime import datetime

logger = logging.getLogger(__name__)

class SmartRetriever:
    def __init__(self, index, embeddings):
        self.index = index
        self.embeddings = embeddings
        self.MAX_DOCUMENTS = 10000

    def get_all_documents(self) -> List[Any]:
        """Recupera tutti i documenti dall'indice."""
        try:
            results = self.index.query(
                vector=[0] * 1536,
                top_k=self.MAX_DOCUMENTS,
                include_metadata=True
            )
            return results.matches
        except Exception as e:
            logger.error(f"Error fetching documents: {str(e)}")
            return []

    def extract_quotes(self, post: Dict) -> List[Dict]:
        """
        Estrae le citazioni dal contenuto raw del post.
        """
        quotes = []
        content = post.get('content', '')
        
        if not isinstance(content, str):
            return quotes

        # Pattern per rilevare le citazioni
        pattern = r"([^:]+?) said:(.*?)Click to expand\.\.\."
        matches = list(re.finditer(pattern, content, re.DOTALL))
        
        # Elabora ogni citazione trovata
        for match in matches:
            quoted_author = match.group(1).strip()
            quoted_text = match.group(2).strip()
            
            if quoted_author and quoted_text:  # Verifica che entrambi i campi siano validi
                quote = {
                    'quoted_author': quoted_author,
                    'quoted_text': quoted_text,
                    'quoting_author': post.get('author'),
                    'post_time': post.get('post_time'),
                    'sentiment': post.get('sentiment', 0),
                    'post_id': post.get('post_id', '')
                }
                quotes.append(quote)

        return quotes

    def process_thread_content(self, thread_posts: List[Dict]) -> Dict:
        """
        Processa il contenuto del thread, organizzando post e citazioni.
        """
        thread_data = {
            'posts': [],
            'quotes': [],
            'total_quotes': 0,
            'quote_network': defaultdict(list),
            'authors': set()
        }
        
        # Prima passata: elabora tutti i post
        for post in thread_posts:
            quotes = self.extract_quotes(post)
            processed_post = {
                'post_id': post.get('post_id', ''),
                'author': post.get('author', ''),
                'content': post.get('content', ''),
                'post_time': post.get('post_time', ''),
                'sentiment': post.get('sentiment', 0),
                'keywords': post.get('keywords', []),
                'quotes': quotes
            }
            
            thread_data['posts'].append(processed_post)
            thread_data['total_quotes'] += len(quotes)
            thread_data['quotes'].extend(quotes)
            thread_data['authors'].add(post.get('author', ''))
            
            # Aggiorna la rete di citazioni
            for quote in quotes:
                thread_data['quote_network'][quote['quoted_author']].append({
                    'quoted_by': quote['quoting_author'],
                    'post_time': quote['post_time'],
                    'sentiment': quote['sentiment']
                })
        
        return thread_data

    def get_thread_analysis(self, matches: List[Any]) -> Dict:
        """Analizza il thread completo."""
        thread_data = defaultdict(lambda: {
            'posts': [],
            'quotes_analysis': {
                'total_quotes': 0,
                'quote_network': defaultdict(list)
            },
            'title': '',
            'url': '',
            'scrape_time': '',
            'metadata': {}
        })
        
        for match in matches:
            metadata = match.metadata
            thread_id = metadata.get('thread_id', 'unknown')
            
            # Costruisci il post dai dati raw
            post = {
                'post_id': metadata.get('post_id', ''),
                'author': metadata.get('author', ''),
                'content': metadata.get('text', ''),  # Usa il contenuto raw
                'post_time': metadata.get('post_time', ''),
                'sentiment': metadata.get('sentiment', 0),
                'keywords': metadata.get('keywords', [])
            }
            
            thread = thread_data[thread_id]
            
            # Aggiorna informazioni base del thread
            if not thread['title']:
                thread['title'] = metadata.get('thread_title', '')
                thread['url'] = metadata.get('url', '')
                thread['scrape_time'] = metadata.get('scrape_time', '')
            
            thread['posts'].append(post)
        
        # Processo ogni thread per estrarre e organizzare le citazioni
        for thread_id, thread in thread_data.items():
            thread_content = self.process_thread_content(thread['posts'])
            thread['quotes_analysis']['total_quotes'] = thread_content['total_quotes']
            thread['quotes_analysis']['quote_network'] = dict(thread_content['quote_network'])
            thread['quotes_analysis']['quotes'] = thread_content['quotes']
        
        return dict(thread_data)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Recupera e analizza i documenti rilevanti."""
        try:
            matches = self.get_all_documents()
            if not matches:
                return [Document(page_content="Nessun documento trovato", metadata={"type": "error"})]
            
            thread_analysis = self.get_thread_analysis(matches)
            
            # Per query specifiche sulle citazioni
            if any(word in query.lower() for word in ['citazioni', 'quote', 'citano']):
                quotes_summary = {}
                for thread_id, thread in thread_analysis.items():
                    quotes_summary[thread_id] = {
                        'total_quotes': thread['quotes_analysis']['total_quotes'],
                        'quotes': [
                            {
                                'quoting_author': q['quoting_author'],
                                'quoted_author': q['quoted_author'],
                                'quoted_text': q['quoted_text'][:100] + '...' if len(q['quoted_text']) > 100 else q['quoted_text']
                            }
                            for q in thread['quotes_analysis'].get('quotes', [])
                        ]
                    }
                
                return [Document(
                    page_content=json.dumps(quotes_summary, ensure_ascii=False, indent=2),
                    metadata={'type': 'quotes_analysis', 'data': quotes_summary}
                )]
            
            # Per altre query
            rich_context = {
                "total_threads": len(thread_analysis),
                "total_posts": sum(len(thread['posts']) for thread in thread_analysis.values()),
                "threads": []
            }
            
            for thread_id, analysis in thread_analysis.items():
                thread_context = {
                    "thread_id": thread_id,
                    "title": analysis['title'],
                    "url": analysis['url'],
                    "scrape_time": analysis['scrape_time'],
                    "total_posts": len(analysis['posts']),
                    "quotes_analysis": analysis['quotes_analysis'],
                    "posts": analysis['posts']
                }
                rich_context["threads"].append(thread_context)
            
            return [Document(
                page_content=json.dumps(rich_context, indent=2, ensure_ascii=False),
                metadata={"type": "rich_analysis", "analysis": rich_context}
            )]
            
        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}", exc_info=True)
            return [Document(
                page_content=f"Errore durante il recupero dei documenti: {str(e)}",
                metadata={"type": "error"}
            )]