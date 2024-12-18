from typing import List, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def validate_post(post: Dict) -> bool:
    """
    Valida la struttura di un post.
    
    Args:
        post: Dizionario contenente i dati del post
        
    Returns:
        bool: True se il post è valido, False altrimenti
    """
    required_fields = ['author', 'post_time', 'content', 'keywords']
    return all(field in post for field in required_fields)

def extract_post_content(post: Dict) -> str:
    """
    Estrae e formatta il contenuto di un post.
    
    Args:
        post: Dizionario contenente i dati del post
        
    Returns:
        str: Contenuto formattato del post
    """
    if not validate_post(post):
        logger.warning(f"Post non valido: {post}")
        return ""
        
    try:
        return f"""
Author: {post['author']}
Time: {post['post_time']}
Content: {post['content']}
Keywords: {', '.join(post['keywords'])}
"""
    except Exception as e:
        logger.error(f"Errore nell'estrazione del contenuto del post: {str(e)}")
        return ""

def process_thread(thread: Dict) -> List[str]:
    """
    Processa un thread e restituisce una lista di contenuti formattati.
    
    Args:
        thread: Dizionario contenente i dati del thread
        
    Returns:
        List[str]: Lista di contenuti formattati
    """
    try:
        texts = []
        
        # Validazione thread
        if not thread.get('title') or not thread.get('url'):
            logger.error(f"Thread non valido: mancano titolo o URL")
            return texts
        
        # Formatta il titolo del thread
        thread_title = f"Thread Title: {thread['title']}\nURL: {thread['url']}\n"
        
        # Processa ogni post
        valid_posts = 0
        for post in thread.get('posts', []):
            content = extract_post_content(post)
            if content:
                full_content = thread_title + content
                texts.append(full_content)
                valid_posts += 1
        
        logger.info(f"Processati {valid_posts} post validi per il thread: {thread['title']}")
        return texts
        
    except Exception as e:
        logger.error(f"Errore nel processamento del thread: {str(e)}", exc_info=True)
        return []