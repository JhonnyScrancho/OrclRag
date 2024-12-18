import streamlit as st

def display_chat_history():
    """Visualizza la cronologia della chat."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def display_chat_input():
    """Gestisce l'input della chat."""
    return st.chat_input("Chiedi all'Oracolo...")

def display_sources(sources):
    """Visualizza le fonti utilizzate."""
    if st.toggle("Mostra fonti"):
        st.divider()
        for doc in sources:
            st.info(doc.metadata.get('thread_title'))
            st.markdown(doc.page_content)