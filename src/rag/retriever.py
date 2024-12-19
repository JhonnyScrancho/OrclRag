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
        self.MAX_DOCUMENTS = 10000  # Aumentato per prendere più documenti

    def get_all_documents(self) -> List[Any]:
        """Recupera tutti i documenti dall'indice."""
        try:
            results = self.index.query(
                vector=[0] * 1536,  # Vector di zeri per recuperare tutto
                top_k=self.MAX_DOCUMENTS,
                include_metadata=True
            )
            return results.matches
        except Exception as e:
            logger.error(f"Error fetching documents: {str(e)}")
            return []

    def extract_quotes(self, content: str) -> List[Dict]:
        """Estrae le citazioni dal contenuto."""
        quotes = []
        if isinstance(content, str):  # Verifica che content sia una stringa
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

    def process_content(self, content: str) -> Dict:
        """Processa il contenuto del post per estrarre informazioni strutturate."""
        structured_content = {}
        if isinstance(content, str):
            lines = content.split('\n')
            current_key = None
            current_value = []
            
            for line in lines:
                if ': ' in line:
                    if current_key:
                        structured_content[current_key] = ' '.join(current_value)
                    key, value = line.split(': ', 1)
                    current_key = key.strip()
                    current_value = [value.strip()]
                elif current_key:
                    current_value.append(line.strip())
            
            if current_key:
                structured_content[current_key] = ' '.join(current_value)
                
        return structured_content

    def get_thread_analysis(self, matches: List[Any]) -> Dict:
        """Analizza in profondità il contenuto del thread."""
        thread_data = defaultdict(lambda: {
            'posts': [],
            'sentiment_trend': [],
            'keywords_frequency': defaultdict(int),
            'authors': set(),
            'conversation_flow': [],
            'title': '',
            'url': '',
            'scrape_time': '',
            'metadata': {}
        })
        
        for match in matches:
            metadata = match.metadata
            thread_id = metadata.get('thread_id', 'unknown')
            content = metadata.get('text', '')
            
            # Processa il contenuto
            structured_content = self.process_content(content)
            
            # Estrai informazioni base del thread
            thread = thread_data[thread_id]
            if not thread['title']:
                thread['title'] = metadata.get('thread_title', 'Unknown Title')
                thread['url'] = metadata.get('url', 'Unknown URL')
                thread['scrape_time'] = metadata.get('scrape_time', '')
            
            # Costruisci il post
            post = {
                'author': structured_content.get('Author', 'Unknown'),
                'time': structured_content.get('Time', 'Unknown'),
                'content': structured_content.get('Content', ''),
                'sentiment': metadata.get('sentiment', 0),
                'keywords': structured_content.get('Keywords', '').split(', ') if structured_content.get('Keywords') else [],
                'quoted_content': self.extract_quotes(structured_content.get('Content', '')),
                'metadata': metadata
            }
            
            # Aggiorna le analisi del thread
            thread['posts'].append(post)
            thread['sentiment_trend'].append(post['sentiment'])
            thread['authors'].add(post['author'])
            
            # Analisi keywords
            for kw in post['keywords']:
                thread['keywords_frequency'][kw] += 1
            
            # Analisi conversazione
            flow_entry = {
                'type': 'reply' if post['quoted_content'] else 'new_topic',
                'from': post['author'],
                'time': post['time'],
                'sentiment': post['sentiment']
            }
            
            if post['quoted_content']:
                flow_entry['to'] = post['quoted_content'][0]['author']
                flow_entry['quoted_content'] = post['quoted_content'][0]['content']
            
            thread['conversation_flow'].append(flow_entry)
                
        return dict(thread_data)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Recupera e analizza i documenti rilevanti."""
        try:
            matches = self.get_all_documents()
            if not matches:
                return [Document(page_content="Nessun documento trovato", metadata={"type": "error"})]
            
            # Analisi approfondita
            thread_analysis = self.get_thread_analysis(matches)
            
            # Crea un documento ricco di contesto
            rich_context = {
                "total_threads": len(thread_analysis),
                "total_posts": sum(len(thread['posts']) for thread in thread_analysis.values()),
                "threads": []
            }
            
            for thread_id, analysis in thread_analysis.items():
                sentiment_trends = analysis['sentiment_trend']
                
                thread_context = {
                    "thread_id": thread_id,
                    "title": analysis['title'],
                    "url": analysis['url'],
                    "scrape_time": analysis['scrape_time'],
                    "total_posts": len(analysis['posts']),
                    "unique_authors": len(analysis['authors']),
                    "sentiment_analysis": {
                        "average": sum(sentiment_trends) / len(sentiment_trends) if sentiment_trends else 0,
                        "trend": sentiment_trends,
                        "variance": statistics.variance(sentiment_trends) if len(sentiment_trends) > 1 else 0
                    },
                    "keywords_analysis": {
                        "top_keywords": sorted(analysis['keywords_frequency'].items(), key=lambda x: x[1], reverse=True)[:10],
                        "total_unique_keywords": len(analysis['keywords_frequency'])
                    },
                    "conversation_dynamics": {
                        "total_replies": sum(1 for flow in analysis['conversation_flow'] if flow['type'] == 'reply'),
                        "new_topics": sum(1 for flow in analysis['conversation_flow'] if flow['type'] == 'new_topic'),
                        "engagement_level": len(analysis['conversation_flow']) / len(analysis['posts']) if analysis['posts'] else 0,
                        "most_active_authors": sorted(
                            [{"author": author, "posts": sum(1 for p in analysis['posts'] if p['author'] == author)} 
                             for author in analysis['authors']], 
                            key=lambda x: x['posts'], 
                            reverse=True
                        )
                    },
                    "posts": [{
                        "author": post['author'],
                        "time": post['time'],
                        "content": post['content'],
                        "sentiment": post['sentiment'],
                        "keywords": post['keywords'],
                        "has_quotes": bool(post['quoted_content']),
                        "quoted_content": post['quoted_content']
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