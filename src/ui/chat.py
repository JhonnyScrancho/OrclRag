import streamlit as st
import logging

logger = logging.getLogger(__name__)

def initialize_chat_state():
    """Inizializza lo stato della chat se non esiste."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def display_chat_history():
    """
    Visualizza la cronologia della chat.
    """
    try:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    except Exception as e:
        logger.error(f"Errore nella visualizzazione della chat history: {str(e)}")
        st.error("Errore nella visualizzazione della cronologia chat")

def display_chat_input():
    """
    Gestisce l'input della chat.
    
    Returns:
        str: Il testo inserito dall'utente, o None se non c'è input
    """
    try:
        return st.chat_input("Chiedi all'Oracolo...")
    except Exception as e:
        logger.error(f"Errore nell'input della chat: {str(e)}")
        return None

def display_sources(sources, show_metadata=True):
    """
    Visualizza le fonti utilizzate.
    
    Args:
        sources: Lista di documenti sorgente
        show_metadata: Se True, mostra i metadata aggiuntivi
    """
    try:
        if st.toggle("Mostra fonti"):
            st.divider()
            for doc in sources:
                with st.expander(f"Fonte: {doc.metadata.get('thread_title', 'Titolo non disponibile')}"):
                    st.markdown(doc.page_content)
                    
                    if show_metadata:
                        st.divider()
                        st.markdown("**Metadata:**")
                        st.markdown(f"- URL: {doc.metadata.get('url', 'Non disponibile')}")
                        st.markdown(f"- Score: {doc.metadata.get('score', 'Non disponibile'):.3f}")
                        st.markdown(f"- Timestamp: {doc.metadata.get('timestamp', 'Non disponibile')}")
                        
    except Exception as e:
        logger.error(f"Errore nella visualizzazione delle fonti: {str(e)}")
        st.error("Errore nella visualizzazione delle fonti")