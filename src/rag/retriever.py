import re
import streamlit as st
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from collections import defaultdict
import logging
import json
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
                vector=[0] * 1536,  # Vector di zeri per recuperare tutti i documenti
                top_k=self.MAX_DOCUMENTS,
                include_metadata=True
            )
            return results.matches
        except Exception as e:
            logger.error(f"Error fetching documents: {str(e)}")
            return []

    def extract_quotes(self, content: str) -> List[Dict]:
        """Estrae le citazioni dal contenuto con pattern migliorato."""
        quotes = []
        if not isinstance(content, str) or not content.strip():
            return quotes
        
        try:
            # Pattern principale per le citazioni nel formato del forum
            main_pattern = r'([^:\n]+?) said:(.*?)(?:(?=\n[^:]*? said:)|$)'
            
            # Debug log del contenuto
            logger.info(f"Analyzing content for quotes: {content[:200]}...")
            
            matches = re.finditer(main_pattern, content, re.DOTALL)
            for match in matches:
                author = match.group(1).strip()
                quoted_text = match.group(2).strip()
                
                # Pulisce il testo citato da "Click to expand..."
                if 'Click to expand' in quoted_text:
                    quoted_text = quoted_text.split('Click to expand')[0].strip()
                
                if author and quoted_text:
                    quote = {
                        'quoted_author': author,
                        'quoted_text': quoted_text,
                        'quote_type': 'explicit'
                    }
                    
                    # Debug log
                    logger.info(f"Found quote from {author}: {quoted_text[:50]}...")
                    
                    # Evita duplicati
                    if not any(q['quoted_text'] == quote['quoted_text'] for q in quotes):
                        quotes.append(quote)
            
            # Log del risultato
            if quotes:
                logger.info(f"Successfully extracted {len(quotes)} quotes")
            else:
                logger.info("No quotes found in content")
            
        except Exception as e:
            logger.error(f"Error extracting quotes: {str(e)}", exc_info=True)
        
        return quotes
    
    def process_content(self, metadata: Dict) -> str:
        """Processa il contenuto del post preservando o ricostruendo il formato delle citazioni."""
        try:
            # Prima controlliamo se abbiamo il testo diretto
            content = metadata.get('text', '')
            if not content:
                logger.warning("No text found in metadata")
                return ""
            
            # Se il contenuto contiene già citazioni nel formato corretto, lo restituiamo
            if ' said:' in content and ('Click to expand' in content or content.count('\n') > 0):
                return content
            
            # Proviamo a ricostruire dalla struttura originale del post
            # Nota: il contenuto originale potrebbe essere nel campo 'content' invece che 'text'
            original_content = metadata.get('content', content)
            
            # Debug log
            logger.info(f"Processing content. Original length: {len(original_content)}")
            logger.info(f"Content starts with: {original_content[:100]}...")
            
            return original_content
            
        except Exception as e:
            logger.error(f"Error in process_content: {str(e)}", exc_info=True)
            return ""

    def build_conversation_context(self, posts: List[Dict]) -> Dict[str, Any]:
        """Costruisce il contesto conversazionale con gestione migliorata delle citazioni."""
        conversation_context = {
            'discussion_threads': [],
            'response_chains': defaultdict(list),
            'topic_context': defaultdict(list),
            'quotes_network': defaultdict(list),
            'temporal_flow': []
        }

        # Indicizza i post per autore per ricerca rapida
        posts_by_author = defaultdict(list)
        for post in posts:
            posts_by_author[post.get('author', '')].append(post)
        
        # Ordina i post per timestamp per analisi temporale
        sorted_posts = sorted(posts, key=lambda x: x.get('time', ''))
        
        for post in sorted_posts:
            quotes = self.extract_quotes(post.get('content', ''))
            post['quotes'] = quotes  # Salva le citazioni nel post
            
            # Aggiungi al flusso temporale
            conversation_context['temporal_flow'].append({
                'time': post.get('time', ''),
                'author': post.get('author', ''),
                'has_quotes': bool(quotes),
                'sentiment': post.get('sentiment', 0),
                'keywords': post.get('keywords', [])
            })
            
            for quote in quotes:
                quoted_author = quote['quoted_author']
                quoted_text = quote['quoted_text']
                
                # Cerca il post originale citato
                original_posts = posts_by_author.get(quoted_author, [])
                original_post = next(
                    (p for p in original_posts if quoted_text in p.get('content', '')),
                    None
                )
                
                if original_post:
                    thread_context = {
                        'original_post': {
                            'author': original_post.get('author', ''),
                            'content': original_post.get('content', ''),
                            'time': original_post.get('time', ''),
                            'sentiment': original_post.get('sentiment', 0)
                        },
                        'response': {
                            'author': post.get('author', ''),
                            'content': post.get('content', ''),
                            'time': post.get('time', ''),
                            'sentiment': post.get('sentiment', 0),
                            'quoted_part': quoted_text
                        },
                        'context_type': 'direct_response',
                        'topic_keywords': list(
                            set(original_post.get('keywords', [])) & 
                            set(post.get('keywords', []))
                        ),
                        'time_delta': self._calculate_time_delta(
                            original_post.get('time', ''),
                            post.get('time', '')
                        )
                    }
                    
                    conversation_context['discussion_threads'].append(thread_context)
                    conversation_context['quotes_network'][quoted_author].append({
                        'responder': post.get('author', ''),
                        'quote': quoted_text,
                        'response': quote.get('response_text', ''),
                        'time': post.get('time', '')
                    })

                    # Aggiungi alla catena di risposte
                    conversation_context['response_chains'][original_post.get('author', '')].append({
                        'responder': post.get('author', ''),
                        'response_type': 'quoted_reply',
                        'time': post.get('time', ''),
                        'sentiment_delta': post.get('sentiment', 0) - original_post.get('sentiment', 0)
                    })

        return conversation_context

    def _calculate_time_delta(self, time1: str, time2: str) -> Optional[float]:
        """Calcola la differenza temporale tra due timestamp."""
        try:
            t1 = datetime.strptime(time1, "%Y-%m-%dT%H:%M:%S%z")
            t2 = datetime.strptime(time2, "%Y-%m-%dT%H:%M:%S%z")
            return (t2 - t1).total_seconds() / 3600  # Ritorna la differenza in ore
        except (ValueError, TypeError):
            return None

    def get_thread_analysis(self, matches: List[Any]) -> Dict:
        """Analisi thread con conteggio citazioni migliorato e metriche avanzate."""
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
                'conversation_threads': [],
                'response_times': [],
                'sentiment_analysis': {
                    'quote_responses': [],
                    'original_posts': []
                }
            },
            'temporal_analysis': {
                'post_frequency': defaultdict(int),
                'activity_hours': defaultdict(int),
                'response_patterns': []
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
            
            # Estrai e analizza le citazioni
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
            
            # Analisi temporale
            try:
                post_time = datetime.strptime(post['time'], "%Y-%m-%dT%H:%M:%S%z")
                thread['temporal_analysis']['post_frequency'][post_time.date().isoformat()] += 1
                thread['temporal_analysis']['activity_hours'][post_time.hour] += 1
            except (ValueError, TypeError):
                pass
            
            # Aggiorna statistiche citazioni e analisi avanzate
            if quotes:
                thread['quotes_analysis']['total_quotes'] += len(quotes)
                for quote in quotes:
                    # Network di citazioni
                    thread['quotes_analysis']['quote_network'][post['author']].append({
                        'quoted_author': quote['quoted_author'],
                        'quoted_text': quote['quoted_text'],
                        'response_text': quote['response_text'],
                        'timestamp': post['time'],
                        'sentiment': post['sentiment']
                    })
                    thread['quotes_analysis']['most_quoted'][quote['quoted_author']] += 1
                    
                    # Analisi del sentiment nelle risposte
                    thread['quotes_analysis']['sentiment_analysis']['quote_responses'].append(post['sentiment'])
            else:
                # Sentiment dei post originali
                thread['quotes_analysis']['sentiment_analysis']['original_posts'].append(post['sentiment'])
            
            # Aggiorna keywords
            for keyword in post['keywords']:
                thread['keywords_frequency'][keyword] += 1
            
            # Metadati thread
            if not thread['title']:
                thread['title'] = metadata.get('thread_title', '')
                thread['url'] = metadata.get('url', '')
                thread['scrape_time'] = metadata.get('timestamp', '')
        
        # Costruisci il contesto conversazionale e analisi finale
        for thread_id, thread in thread_data.items():
            # Analisi conversazionale
            thread['conversation_context'] = self.build_conversation_context(thread['posts'])
            
            # Calcola metriche aggiuntive
            if thread['quotes_analysis']['sentiment_analysis']['quote_responses']:
                thread['quotes_analysis']['sentiment_analysis']['avg_response_sentiment'] = (
                    sum(thread['quotes_analysis']['sentiment_analysis']['quote_responses']) /
                    len(thread['quotes_analysis']['sentiment_analysis']['quote_responses'])
                )
            
            if thread['quotes_analysis']['sentiment_analysis']['original_posts']:
                thread['quotes_analysis']['sentiment_analysis']['avg_original_sentiment'] = (
                    sum(thread['quotes_analysis']['sentiment_analysis']['original_posts']) /
                    len(thread['quotes_analysis']['sentiment_analysis']['original_posts'])
                )
            
            # Ordina e limita le keywords più frequenti
            thread['top_keywords'] = sorted(
                thread['keywords_frequency'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
        return dict(thread_data)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Recupera documenti con contesto conversazionale arricchito."""
        try:
            matches = self.get_all_documents()
            if not matches:
                return [Document(page_content="Nessun documento trovato", metadata={"type": "error"})]
            
            # Processa ogni documento per estrarre citazioni
            for match in matches:
                content = self.process_content(match.metadata)
                quotes = self.extract_quotes(content)
                match.metadata['quotes'] = quotes
                match.metadata['quotes_count'] = len(quotes)
            
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
                        "total_quotes": analysis['quotes_analysis']['total_quotes'],
                        "conversation_threads": analysis['quotes_analysis'].get('conversation_threads', []),
                        "most_quoted_authors": sorted(
                            analysis['quotes_analysis']['most_quoted'].items(),
                            key=lambda x: x[1],
                            reverse=True
                        )
                    },
                    "temporal_analysis": analysis['temporal_analysis'],
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