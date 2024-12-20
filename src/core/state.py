import streamlit as st
from pinecone import Pinecone
from config import INDEX_NAME
import logging

logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'processed_threads' not in st.session_state:
        st.session_state.processed_threads = set()
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "chat"

def initialize_pinecone():
    """Initialize Pinecone connection"""
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        
        # Check if index exists
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if INDEX_NAME not in existing_indexes:
            error_msg = f"Index {INDEX_NAME} does not exist. Available indexes: {', '.join(existing_indexes) if existing_indexes else 'none'}"
            st.error(error_msg)
            return None
        
        # Get the index
        index = pc.Index(INDEX_NAME)
        
        # Verify index is not empty
        stats = index.describe_index_stats()
        if stats['total_vector_count'] == 0:
            st.warning("Database is empty. Please load data from the 'Database' tab.")
            
        return index
        
    except Exception as e:
        error_msg = f"""
        Error connecting to Pinecone:
        - Error: {str(e)}
        - API Key length: {len(st.secrets['PINECONE_API_KEY'])}
        - Requested index: {INDEX_NAME}
        """
        logger.error(error_msg)
        st.error(error_msg)
        return None