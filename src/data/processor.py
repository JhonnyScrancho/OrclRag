from typing import List, Dict
from datetime import datetime
import hashlib
import re
import logging

logger = logging.getLogger(__name__)

def generate_post_id(post: Dict, thread_id: str) -> str:
    """Genera un ID unico per ogni post basato sul suo contenuto e timestamp."""
    post_key = f"{thread_id}_{post['post_id']}_{post['post_time']}"
    return hashlib.md5(post_key.encode()).hexdigest()

def extract_quote(content: str) -> tuple[str, str]:
    """Estrae la citazione e il contenuto effettivo dal post."""
    # Pattern per identificare le citazioni
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

def extract_post_content(post: Dict, thread_id: str) -> dict:
    """Estrae e formatta il contenuto di un post con metadati estesi."""
    try:
        # Standardizza il timestamp
        post_time = datetime.strptime(post['post_time'], "%Y-%m-%dT%H:%M:%S%z").isoformat()
    except ValueError:
        post_time = post['post_time']
    
    # Estrai citazione e contenuto effettivo
    quote_info, actual_content = extract_quote(post['content'])
    
    # Costruisci il testo formattato
    formatted_text = f"""
Author: {post['author']}
Time: {post_time}
"""
    
    if quote_info:
        formatted_text += f"""
Quoted Author: {quote_info['quoted_author']}
Quoted Content: {quote_info['quoted_content']}
"""
    
    formatted_text += f"""
Content: {actual_content}
Keywords: {', '.join(post['keywords'])}
Sentiment: {post.get('sentiment', 0)}
"""
    
    metadata = {
        "post_id": post['post_id'],
        "unique_post_id": generate_post_id(post, thread_id),
        "author": post['author'],
        "post_time": post_time,
        "keywords": post['keywords'],
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

def process_thread(thread: Dict) -> List[str]:
    """Processa un thread e restituisce una lista di testi per il chunking."""
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
        "actual_posts": actual_posts,  # Numero effettivo di post
        "declared_posts": declared_posts,  # Numero dichiarato di post
        "is_thread": True
    }
    
    for post in thread['posts']:
        metadata = extract_post_content(post, thread_id)
        metadata.update(thread_metadata)
        metadata["is_chunk"] = False  # Indica che questo è un post reale, non solo un chunk
        processed_posts.append(metadata["text"])
    
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
        set(existing_post.get('keywords', [])) != set(new_post.get('keywords', []))
    )

async def update_thread_in_index(index, thread: Dict, embeddings):
    """Aggiorna un thread nell'indice, gestendo i duplicati e gli aggiornamenti."""
    thread_id = get_thread_id(thread)
    
    # Metadati comuni del thread
    thread_metadata = {
        "thread_id": thread_id,
        "thread_title": thread['title'],
        "url": thread['url'],
        "scrape_time": thread['scrape_time'],
        "total_posts": len(thread['posts']),
        "is_thread": True
    }
    
    if 'metadata' in thread:
        thread_metadata.update(thread['metadata'])
    
    # Recupera i post esistenti per questo thread
    existing_posts = await index.query(
        filter={"thread_id": thread_id},
        include_metadata=True
    )
    
    existing_post_ids = {post.metadata.get('unique_post_id'): post for post in existing_posts.matches}
    
    # Processa ogni post
    for post in thread['posts']:
        metadata = extract_post_content(post, thread_id)
        metadata.update(thread_metadata)
        metadata["is_post"] = True
        post_id = metadata['unique_post_id']
        
        if post_id not in existing_post_ids:
            # Nuovo post - aggiungi all'indice
            embedding = embeddings.embed_query(metadata['text'])
            await index.upsert(
                vectors=[{
                    "id": post_id,
                    "values": embedding,
                    "metadata": metadata
                }]
            )
        else:
            # Post esistente - verifica se necessita aggiornamento
            existing_post = existing_post_ids[post_id]
            if should_update_post(existing_post.metadata, metadata):
                embedding = embeddings.embed_query(metadata['text'])
                await index.upsert(
                    vectors=[{
                        "id": post_id,
                        "values": embedding,
                        "metadata": metadata
                    }]
                )

    return len(thread['posts'])