import streamlit as st
from datetime import datetime

def display_thread_preview(thread):
    """Mostra un'anteprima del thread."""
    st.subheader(thread['title'])
    st.write(f"URL: {thread['url']}")
    st.write(f"Data scraping: {thread['scrape_time']}")
    
    # Rimosso l'expander e sostituito con una divisione visiva
    st.write("Posts:")
    for post in thread['posts']:
        st.markdown(f"""
        **Autore:** {post['author']}  
        **Data:** {post['post_time']}  
        **Contenuto:** {post['content']}  
        **Keywords:** {', '.join(post['keywords'])}
        ---
        """)