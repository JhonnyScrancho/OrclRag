from typing import List, Dict
from datetime import datetime
import hashlib

def generate_post_id(post: Dict, thread_id: str) -> str:
    """Genera un ID unico per ogni post basato sul suo contenuto e timestamp."""
    post_key = f"{thread_id}_{post['post_id']}_{post['post_time']}"
    return hashlib.md5(post_key.encode()).hexdigest()

def extract_post_content(post: Dict, thread_id: str) -> dict:
    """Estrae e formatta il contenuto di un post con metadati estesi."""
    try:
        # Standardizza il timestamp
        post_time = datetime.strptime(post['post_time'], "%Y-%m-%dT%H:%M:%S%z").isoformat()
    except ValueError:
        post_time = post['post_time']  # Mantieni il formato originale se il parsing fallisce
    
    return {
        "text": f"""
Author: {post['author']}
Time: {post_time}
Content: {post['content']}
Keywords: {', '.join(post['keywords'])}
""",
        "metadata": {
            "post_id": post['post_id'],
            "unique_post_id": generate_post_id(post, thread_id),
            "author": post['author'],
            "post_time": post_time,
            "keywords": post['keywords'],
            "sentiment": post.get('sentiment', 0),
            "content_length": len(post['content']),
            "thread_id": thread_id
        }
    }

def process_thread(thread: Dict) -> List[dict]:
    """Processa un thread e restituisce una lista di contenuti formattati con metadati."""
    processed_posts = []
    thread_id = get_thread_id(thread)
    
    # Metadati comuni del thread
    thread_metadata = {
        "thread_id": thread_id,
        "thread_title": thread['title'],
        "url": thread['url'],
        "scrape_time": thread['scrape_time'],
        "total_posts": len(thread['posts'])
    }
    
    # Processa ogni post
    for post in thread['posts']:
        processed_post = extract_post_content(post, thread_id)
        # Aggiungi i metadati del thread
        processed_post['metadata'].update(thread_metadata)
        processed_posts.append(processed_post)
    
    return processed_posts

def get_thread_id(thread: Dict) -> str:
    """Genera un ID unico per il thread."""
    thread_key = f"{thread['url']}_{thread['scrape_time']}"
    return hashlib.md5(thread_key.encode()).hexdigest()

def should_update_post(existing_post: Dict, new_post: Dict) -> bool:
    """Determina se un post esistente dovrebbe essere aggiornato."""
    # Verifica se ci sono modifiche significative
    return (
        existing_post.get('content_length') != new_post.get('content_length') or
        existing_post.get('sentiment') != new_post.get('sentiment') or
        set(existing_post.get('keywords', [])) != set(new_post.get('keywords', []))
    )

async def update_thread_in_index(index, thread: Dict, embeddings):
    """Aggiorna un thread nell'indice, gestendo i duplicati e gli aggiornamenti."""
    processed_posts = process_thread(thread)
    thread_id = get_thread_id(thread)
    
    # Recupera i post esistenti per questo thread
    existing_posts = await index.query(
        filter={"thread_id": thread_id},
        include_metadata=True
    )
    
    existing_post_ids = {post.metadata.get('unique_post_id'): post for post in existing_posts.matches}
    
    for processed_post in processed_posts:
        post_id = processed_post['metadata']['unique_post_id']
        
        if post_id not in existing_post_ids:
            # Nuovo post - aggiungi all'indice
            embedding = embeddings.embed_query(processed_post['text'])
            await index.upsert(
                vectors=[{
                    "id": post_id,
                    "values": embedding,
                    "metadata": processed_post['metadata']
                }]
            )
        else:
            # Post esistente - verifica se necessita aggiornamento
            existing_post = existing_post_ids[post_id]
            if should_update_post(existing_post.metadata, processed_post['metadata']):
                embedding = embeddings.embed_query(processed_post['text'])
                await index.upsert(
                    vectors=[{
                        "id": post_id,
                        "values": embedding,
                        "metadata": processed_post['metadata']
                    }]
                )

    return len(processed_posts)