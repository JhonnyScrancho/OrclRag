from typing import List, Dict, Tuple, Optional
from datetime import datetime
import hashlib
import re
import logging
import time
import asyncio

logger = logging.getLogger(__name__)

def generate_post_id(post: Dict, thread_id: str) -> str:
    """Genera un ID unico per ogni post basato sul suo contenuto e timestamp."""
    post_key = f"{thread_id}_{post['post_id']}_{post['post_time']}"
    return hashlib.md5(post_key.encode()).hexdigest()

def extract_quote(content: str) -> tuple[Optional[Dict], str]:
    """Estrae la citazione e il contenuto effettivo dal post."""
    quote_pattern = r"(.*?) said:(.*?)Click to expand\.\.\.(.*)"
    match = re.search(quote_pattern, content)
    
    if match:
        quoted_author = match.group(1).strip()
        quoted_content = match.group(2).strip()
        actual_content = match.group(3).strip()
        
        quote_info = {
            "quoted_author": quoted_author,
            "quoted_content": quoted_content
        }
        
        return quote_info, actual_content
    
    return None, content

def extract_post_content(post: Dict, thread_id: str) -> Dict:
    """Estrae e formatta il contenuto di un post con metadati estesi."""
    try:
        # Standardizza il timestamp
        post_time = datetime.strptime(post['post_time'], "%Y-%m-%dT%H:%M:%S%z").isoformat()
    except ValueError:
        post_time = post['post_time']
    
    # Estrai citazione e contenuto effettivo
    quote_info, actual_content = extract_quote(post['content'])
    
    # Costruisci il testo formattato
    formatted_text = f"""Author: {post['author']}
Time: {post_time}
"""
    
    if quote_info:
        formatted_text += f"""
Quoted Author: {quote_info['quoted_author']}
Quoted Content: {quote_info['quoted_content']}
"""
    
    formatted_text += f"""
Content: {actual_content}
Keywords: {', '.join(post.get('keywords', []))}
Sentiment: {post.get('sentiment', 0)}
"""
    
    metadata = {
        "post_id": post['post_id'],
        "unique_post_id": generate_post_id(post, thread_id),
        "author": post['author'],
        "post_time": post_time,
        "keywords": post.get('keywords', []),
        "sentiment": post.get('sentiment', 0),
        "content_length": len(actual_content),
        "thread_id": thread_id,
        "text": formatted_text
    }
    
    if quote_info:
        metadata["quoted_author"] = quote_info["quoted_author"]
        metadata["quoted_content"] = quote_info["quoted_content"]
    
    # Aggiungi eventuali metadati aggiuntivi dal post originale
    if 'metadata' in post:
        metadata.update(post['metadata'])
    
    return metadata

def process_thread(thread: Dict) -> List[Dict]:
    """Processa un thread e restituisce una lista di post processati."""
    thread_id = get_thread_id(thread)
    processed_posts = []
    
    # Verifica e log del numero di posts
    actual_posts = len(thread['posts'])
    declared_posts = thread.get('metadata', {}).get('total_posts', 0)
    logger.info(f"Processing thread {thread_id}: {actual_posts} actual posts, {declared_posts} declared posts")
    
    # Metadati comuni del thread
    thread_metadata = {
        "thread_id": thread_id,
        "thread_title": thread['title'],
        "url": thread['url'],
        "scrape_time": thread['scrape_time'],
        "actual_posts": actual_posts,
        "declared_posts": declared_posts,
        "is_thread": True
    }
    
    seen_post_ids = set()  # Per evitare duplicati
    
    for post in thread['posts']:
        try:
            # Verifica duplicati
            post_id = post['post_id']
            if post_id in seen_post_ids:
                logger.warning(f"Duplicate post found: {post_id}")
                continue
            
            seen_post_ids.add(post_id)
            
            # Estrai metadata e contenuto
            metadata = extract_post_content(post, thread_id)
            metadata.update(thread_metadata)
            
            # Aggiungi il post processato
            processed_posts.append({
                "metadata": metadata,
                "text": metadata["text"]
            })
            
        except Exception as e:
            logger.error(f"Error processing post {post.get('post_id', 'unknown')}: {str(e)}")
            continue
    
    logger.info(f"Successfully processed {len(processed_posts)} posts from thread {thread_id}")
    return processed_posts

def get_thread_id(thread: Dict) -> str:
    """Genera un ID unico per il thread."""
    thread_key = f"{thread['url']}_{thread['scrape_time']}"
    return hashlib.md5(thread_key.encode()).hexdigest()

def should_update_post(existing_post: Dict, new_post: Dict) -> bool:
    """Determina se un post esistente dovrebbe essere aggiornato."""
    return (
        existing_post.get('content_length') != new_post.get('content_length') or
        existing_post.get('sentiment') != new_post.get('sentiment') or
        set(existing_post.get('keywords', [])) != set(new_post.get('keywords', [])) or
        existing_post.get('last_updated', '') < new_post.get('last_updated', '')
    )

async def update_thread_in_index(index, thread: Dict, embeddings):
    """Aggiorna un thread nell'indice, gestendo i duplicati e gli aggiornamenti."""
    thread_id = get_thread_id(thread)
    updates = 0
    
    try:
        # Processa il thread
        processed_posts = process_thread(thread)
        
        # Recupera i post esistenti
        existing_posts = await index.query(
            filter={"thread_id": thread_id},
            include_metadata=True
        )
        
        existing_post_map = {
            post.metadata.get('post_id'): post 
            for post in existing_posts.matches
        }
        
        # Aggiorna o inserisci ogni post
        for post in processed_posts:
            post_id = post['metadata']['post_id']
            
            if post_id in existing_post_map:
                existing = existing_post_map[post_id]
                if should_update_post(existing.metadata, post['metadata']):
                    vector = embeddings.embed_query(post['text'])
                    await index.upsert([(post_id, vector, post['metadata'])])
                    updates += 1
            else:
                vector = embeddings.embed_query(post['text'])
                await index.upsert([(post_id, vector, post['metadata'])])
                updates += 1
        
        logger.info(f"Updated {updates} posts in thread {thread_id}")
        return updates
        
    except Exception as e:
        logger.error(f"Error updating thread {thread_id}: {str(e)}")
        raise

def update_document_in_index(index, doc_id: str, embedding: List[float], metadata: Dict):
    """Aggiorna un singolo documento nell'indice."""
    try:
        # Verifica dimensione embedding
        if len(embedding) != len(embedding):
            raise ValueError(f"Invalid embedding dimension: {len(embedding)}")
        
        index.upsert(
            vectors=[{
                "id": doc_id,
                "values": embedding,
                "metadata": metadata
            }]
        )
        
        logger.info(f"Successfully updated document {doc_id}")
        
    except Exception as e:
        logger.error(f"Error updating document {doc_id}: {str(e)}")
        raise

def validate_metadata(metadata: Dict) -> bool:
    """Valida i metadati di un post."""
    required_fields = ['post_id', 'thread_id', 'author', 'post_time']
    return all(field in metadata for field in required_fields)

def sanitize_content(content: str) -> str:
    """Pulisce il contenuto del post."""
    # Rimuovi caratteri non validi
    content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
    # Normalizza spazi
    content = ' '.join(content.split())
    return content