import streamlit as st
from ui.styles import apply_custom_styles
from ui.views import render_chat_interface, render_database_view, render_settings
from ui.components import render_sidebar
from core.state import initialize_session_state, initialize_pinecone
from embeddings.generator import get_embeddings
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main application with improved navigation and UI"""
    initialize_session_state()
    apply_custom_styles()
    
    try:
        index = initialize_pinecone()
        if index is None:
            st.stop()
        
        embeddings = get_embeddings()
        
        # Render sidebar
        render_sidebar()
        
        # Render main content based on navigation
        if st.session_state.current_page == "chat":
            render_chat_interface(index, embeddings)
        elif st.session_state.current_page == "database":
            render_database_view(index)
        elif st.session_state.current_page == "settings":
            render_settings()
    
    except Exception as e:
        st.error(f"Application Error: {str(e)}")

if __name__ == "__main__":
    main()