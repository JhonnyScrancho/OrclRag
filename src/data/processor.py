from typing import List, Dict
from datetime import datetime

def extract_post_content(post: Dict) -> str:
    """Estrae e formatta il contenuto di un post."""
    return f"""
Author: {post['author']}
Time: {post['post_time']}
Content: {post['content']}
Keywords: {', '.join(post['keywords'])}
"""

def process_thread(thread: Dict) -> List[str]:
    """Processa un thread e restituisce una lista di contenuti formattati."""
    texts = []
    
    # Aggiungi il titolo del thread
    thread_title = f"Thread Title: {thread['title']}\nURL: {thread['url']}\n"
    
    # Processa ogni post
    for post in thread['posts']:
        full_content = thread_title + extract_post_content(post)
        texts.append(full_content)
    
    return texts