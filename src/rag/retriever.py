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
        Estrae le citazioni dal contenuto di un post.
        Ora gestisce correttamente tutte le varianti di citazione presenti nel contenuto.
        """
        quotes = []
        content = post.get('content', '')
        
        if not isinstance(content, str):
            return quotes

        # Pattern principale per le citazioni (cattura l'intera struttura)
        pattern = r"([\w\s]+) said:(.*?)Click to expand\.\.\.(.*)"
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            quoted_author = match.group(1).strip()
            quoted_text = match.group(2).strip()
            response_text = match.group(3).strip() if match.group(3) else ""
            
            if quoted_author and quoted_text:
                quotes.append({
                    'quote_id': f"{post.get('post_id')}_{len(quotes)}",
                    'quoted_author': quoted_author,
                    'quoted_text': quoted_text,
                    'response_text': response_text,
                    'quoting_author': post.get('author'),
                    'post_time': post.get('post_time'),
                    'post_id': post.get('post_id'),
                    'sentiment': post.get('sentiment', 0),
                    'keywords': post.get('keywords', [])
                })
        
        return quotes

    def build_conversation_tree(self, posts: List[Dict]) -> Dict:
        """
        Costruisce un albero della conversazione basato sulle citazioni.
        """
        conversation_tree = {
            'posts_by_id': {},
            'citation_chains': [],
            'author_interactions': defaultdict(list)
        }

        # Prima passata: organizza i post per ID e estrai le citazioni
        for post in posts:
            post_id = post.get('post_id')
            quotes = self.extract_quotes(post)
            
            conversation_tree['posts_by_id'][post_id] = {
                'post': post,
                'quotes': quotes,
                'replies': []
            }

            # Traccia le interazioni tra autori
            for quote in quotes:
                conversation_tree['author_interactions'][post['author']].append({
                    'interacts_with': quote['quoted_author'],
                    'interaction_type': 'quote',
                    'timestamp': post['post_time']
                })

        # Seconda passata: costruisci le catene di citazioni
        for post_id, post_data in conversation_tree['posts_by_id'].items():
            for quote in post_data['quotes']:
                # Trova il post originale citato
                for original_id, original_data in conversation_tree['posts_by_id'].items():
                    original_post = original_data['post']
                    if (original_post['author'] == quote['quoted_author'] and 
                        quote['quoted_text'] in original_post['content']):
                        # Aggiungi alla catena di citazioni
                        conversation_tree['citation_chains'].append({
                            'original_post_id': original_id,
                            'quoting_post_id': post_id,
                            'quote_data': quote
                        })
                        # Aggiungi alla lista di risposte del post originale
                        original_data['replies'].append(post_id)

        return conversation_tree

    def get_thread_analysis(self, matches: List[Any]) -> Dict:
        """Analizza in profondità il contenuto del thread."""
        thread_data = defaultdict(lambda: {
            'posts': [],
            'sentiment_trend': [],
            'keywords_frequency': defaultdict(int),
            'authors': set(),
            'conversation_tree': None,
            'quotes_analysis': {
                'total_quotes': 0,
                'quotes_by_author': defaultdict(list),
                'most_quoted_authors': defaultdict(int)
            },
            'title': '',
            'url': '',
            'scrape_time': '',
            'metadata': {}
        })
        
        for match in matches:
            metadata = match.metadata
            thread_id = metadata.get('thread_id', 'unknown')
            
            # Costruisci il post con tutti i metadati necessari
            post = {
                'post_id': metadata.get('post_id', f"post_{len(thread_data[thread_id]['posts'])}"),
                'author': metadata.get('author', 'Unknown'),
                'content': metadata.get('text', ''),
                'post_time': metadata.get('post_time', 'Unknown'),
                'sentiment': metadata.get('sentiment', 0),
                'keywords': metadata.get('keywords', []),
            }
            
            thread = thread_data[thread_id]
            thread['posts'].append(post)
            
            # Aggiorna metadati thread
            if not thread['title']:
                thread['title'] = metadata.get('thread_title', 'Unknown Title')
                thread['url'] = metadata.get('url', 'Unknown URL')
                thread['scrape_time'] = metadata.get('scrape_time', '')
            
            # Aggiorna statistiche
            thread['sentiment_trend'].append(post['sentiment'])
            thread['authors'].add(post['author'])
            
            for kw in post['keywords']:
                thread['keywords_frequency'][kw] += 1
            
        # Costruisci l'albero della conversazione per ogni thread
        for thread_id, thread in thread_data.items():
            conversation_tree = self.build_conversation_tree(thread['posts'])
            thread['conversation_tree'] = conversation_tree
            
            # Aggiorna analisi citazioni
            for chain in conversation_tree['citation_chains']:
                quote_data = chain['quote_data']
                thread['quotes_analysis']['total_quotes'] += 1
                thread['quotes_analysis']['quotes_by_author'][quote_data['quoting_author']].append(quote_data)
                thread['quotes_analysis']['most_quoted_authors'][quote_data['quoted_author']] += 1
            
        return dict(thread_data)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Recupera e analizza i documenti rilevanti."""
        try:
            matches = self.get_all_documents()
            if not matches:
                return [Document(page_content="Nessun documento trovato", metadata={"type": "error"})]
            
            thread_analysis = self.get_thread_analysis(matches)
            
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
                    "unique_authors": len(analysis['authors']),
                    "total_quotes": analysis['quotes_analysis']['total_quotes'],
                    "conversation_tree": analysis['conversation_tree'],
                    "sentiment_analysis": {
                        "average": sum(analysis['sentiment_trend']) / len(analysis['sentiment_trend']) if analysis['sentiment_trend'] else 0,
                        "trend": analysis['sentiment_trend']
                    },
                    "interaction_analysis": {
                        "most_active_quoters": sorted(
                            [(author, len(quotes)) for author, quotes in analysis['quotes_analysis']['quotes_by_author'].items()],
                            key=lambda x: x[1],
                            reverse=True
                        ),
                        "most_quoted": sorted(
                            analysis['quotes_analysis']['most_quoted_authors'].items(),
                            key=lambda x: x[1],
                            reverse=True
                        )
                    },
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