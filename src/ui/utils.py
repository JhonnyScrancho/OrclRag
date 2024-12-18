import streamlit as st
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def format_timestamp(timestamp: str) -> str:
    """
    Formatta il timestamp in un formato leggibile.
    
    Args:
        timestamp: Timestamp in formato stringa
        
    Returns:
        str: Timestamp formattato
    """
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime("%d/%m/%Y %H:%M")
    except Exception as e:
        logger.warning(f"Errore nella formattazione del timestamp: {str(e)}")
        return timestamp

def display_thread_preview(thread: dict):
    """
    Mostra un'anteprima del thread.
    
    Args:
        thread: Dizionario contenente i dati del thread
    """
    try:
        st.subheader(thread.get('title', 'Titolo non disponibile'))
        st.write(f"URL: {thread.get('url', 'URL non disponibile')}")
        
        scrape_time = thread.get('scrape_time')
        if scrape_time:
            formatted_time = format_timestamp(scrape_time)
            st.write(f"Data scraping: {formatted_time}")
        
        with st.expander("Posts"):
            posts = thread.get('posts', [])
            if not posts:
                st.warning("Nessun post disponibile")
                return
                
            for post in posts:
                st.markdown(f"""
                **Autore:** {post.get('author', 'Anonimo')}  
                **Data:** {format_timestamp(post.get('post_time', ''))}  
                **Contenuto:** {post.get('content', 'Contenuto non disponibile')}  
                **Keywords:** {', '.join(post.get('keywords', []))}
                ---
                """)
                
    except Exception as e:
        logger.error(f"Errore nella visualizzazione del thread: {str(e)}", exc_info=True)
        st.error("Errore nella visualizzazione del thread")