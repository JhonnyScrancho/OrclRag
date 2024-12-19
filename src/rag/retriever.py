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

    def extract_quotes(self, content: str) -> List[Dict]:
        """Estrae le citazioni dal contenuto con contesto conversazionale."""
        quotes = []
        if not isinstance(content, str):
            return quotes
            
        # Pattern migliorato per le citazioni con cattura del contesto
        pattern = r"(.*?) said:(.*?)Click to expand\.\.\.(.*)"
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            author = match.group(1).strip()
            quoted_text = match.group(2).strip()
            response_text = match.group(3).strip() if match.group(3) else ""
            
            if author and quoted_text:
                quotes.append({
                    'quoted_author': author,
                    'quoted_text': quoted_text,
                    'response_text': response_text,
                    'quote_type': 'explicit',
                    'original_text': match.group(0),
                    'context_relation': 'direct_response'  # Indica una risposta diretta al post citato
                })
        
        return quotes

    def build_conversation_context(self, posts: List[Dict]) -> Dict[str, Any]:
        """Costruisce il contesto conversazionale basato sulle citazioni."""
        conversation_context = {
            'discussion_threads': [],  # Lista di thread di discussione collegati
            'response_chains': defaultdict(list),  # Catene di risposta
            'topic_context': defaultdict(list)  # Contesto per argomento
        }

        # Traccia le relazioni tra post
        post_relations = defaultdict(list)
        
        for post in posts:
            post_id = post.get('post_id', '')
            quotes = post.get('quotes', [])
            
            for quote in quotes:
                quoted_author = quote['quoted_author']
                quoted_text = quote['quoted_text']
                
                # Trova il post originale citato
                original_post = next(
                    (p for p in posts if p['author'] == quoted_author and quoted_text in p.get('content', '')),
                    None
                )
                
                if original_post:
                    # Aggiungi al contesto conversazionale
                    thread_context = {
                        'original_post': {
                            'author': original_post['author'],
                            'content': original_post['content'],
                            'time': original_post['time'],
                            'sentiment': original_post['sentiment']
                        },
                        'response': {
                            'author': post['author'],
                            'content': post['content'],
                            'time': post['time'],
                            'sentiment': post['sentiment'],
                            'quoted_part': quoted_text
                        },
                        'context_type': 'direct_response',
                        'topic_keywords': list(set(original_post['keywords']).intersection(post['keywords']))
                    }
                    
                    conversation_context['discussion_threads'].append(thread_context)
                    conversation_context['response_chains'][original_post['author']].append({
                        'responder': post['author'],
                        'response_type': 'quoted_reply',
                        'time': post['time']
                    })

        return conversation_context

    def get_thread_analysis(self, matches: List[Any]) -> Dict:
        """Analizza il thread con contesto conversazionale avanzato."""
        thread_data = defaultdict(lambda: {
            'posts': [],
            'conversation_context': None,
            'sentiment_trend': [],
            'keywords_frequency': defaultdict(int),
            'authors': set(),
            'conversation_flow': [],
            'quotes_analysis': {
                'total_quotes': 0,
                'quote_network': defaultdict(list),
                'most_quoted': defaultdict(int),
                'conversation_threads': []
            },
            'title': '',
            'url': '',
            'scrape_time': '',
            'metadata': {}
        })
        
        for match in matches:
            metadata = match.metadata
            thread_id = metadata.get('thread_id', 'unknown')
            content = metadata.get('text', '')
            
            # Estrai citazioni e costruisci il post
            quotes = self.extract_quotes(content)
            post = {
                'post_id': metadata.get('post_id', ''),
                'author': metadata.get('author', 'Unknown'),
                'time': metadata.get('post_time', 'Unknown'),
                'content': content,
                'sentiment': metadata.get('sentiment', 0),
                'keywords': metadata.get('keywords', []),
                'quotes': quotes,
                'has_quotes': bool(quotes)
            }
            
            thread = thread_data[thread_id]
            thread['posts'].append(post)
            thread['sentiment_trend'].append(post['sentiment'])
            thread['authors'].add(post['author'])
            
            # Aggiorna statistiche citazioni e contesto
            if quotes:
                thread['quotes_analysis']['total_quotes'] += len(quotes)
                for quote in quotes:
                    thread['quotes_analysis']['quote_network'][post['author']].append({
                        'quoted_author': quote['quoted_author'],
                        'quoted_text': quote['quoted_text'],
                        'response_text': quote['response_text'],
                        'timestamp': post['time'],
                        'context_relation': quote['context_relation']
                    })
                    thread['quotes_analysis']['most_quoted'][quote['quoted_author']] += 1
            
            # Analisi keywords nel contesto
            for kw in post['keywords']:
                thread['keywords_frequency'][kw] += 1
        
        # Costruisci il contesto conversazionale per ogni thread
        for thread_id, thread in thread_data.items():
            thread['conversation_context'] = self.build_conversation_context(thread['posts'])
        
        return dict(thread_data)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Recupera documenti con contesto conversazionale arricchito."""
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
                    "conversation_context": analysis['conversation_context'],
                    "quotes_analysis": analysis['quotes_analysis'],
                    "sentiment_analysis": {
                        "average": sum(analysis['sentiment_trend']) / len(analysis['sentiment_trend']) if analysis['sentiment_trend'] else 0,
                        "trend": analysis['sentiment_trend']
                    },
                    "keywords_analysis": {
                        "top_keywords": sorted(analysis['keywords_frequency'].items(), key=lambda x: x[1], reverse=True)[:10],
                        "total_unique_keywords": len(analysis['keywords_frequency'])
                    },
                    "conversation_dynamics": {
                        "total_replies": sum(1 for flow in analysis['conversation_flow'] if flow['type'] == 'reply'),
                        "new_topics": sum(1 for flow in analysis['conversation_flow'] if flow['type'] == 'new_topic'),
                        "total_quotes": analysis['quotes_analysis']['total_quotes'],
                        "conversation_threads": analysis['quotes_analysis']['conversation_threads'],
                        "most_quoted_authors": sorted(
                            analysis['quotes_analysis']['most_quoted'].items(),
                            key=lambda x: x[1],
                            reverse=True
                        )
                    },
                    "posts": [{
                        "author": post['author'],
                        "time": post['time'],
                        "content": post['content'],
                        "sentiment": post['sentiment'],
                        "keywords": post['keywords'],
                        "quotes": post['quotes'],
                        "has_quotes": post['has_quotes']
                    } for post in analysis['posts']]
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