import streamlit as st
from config import INDEX_NAME, LLM_MODEL 
from data.loader import load_json
from data.processor import process_thread
from embeddings.generator import create_chunks, get_embeddings
from embeddings.indexer import update_document_in_index
from rag.retriever import SmartRetriever
from rag.chain import setup_rag_chain
import hashlib
import time
from datetime import datetime
from pinecone import Pinecone
from ui.styles import apply_custom_styles, render_sidebar
from config import INDEX_NAME
import pandas as pd
import logging
from functools import wraps
from collections import deque
from config import MAX_REQUESTS_PER_MINUTE, RATE_LIMIT_PERIOD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="🔮 L'Oracolo",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'processed_threads' not in st.session_state:
        st.session_state.processed_threads = set()

def get_thread_id(thread):
    """Genera un ID unico per il thread."""
    thread_key = f"{thread['url']}_{thread['scrape_time']}"
    return hashlib.md5(thread_key.encode()).hexdigest()

def initialize_pinecone():
    """Inizializza connessione a Pinecone."""
    try:
        pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        index = pc.Index(INDEX_NAME)
        
        # Verifica che l'indice contenga dati
        stats = index.describe_index_stats()
        if stats['total_vector_count'] == 0:
            st.warning("Il database è vuoto. Carica dei dati dalla tab 'Caricamento'.")
            
        return index
    except Exception as e:
        st.error(f"Errore connessione Pinecone: {str(e)}")
        return None

def process_and_index_thread(thread, embeddings, index):
    """Processa e indicizza un thread."""
    thread_id = get_thread_id(thread)
    
    try:
        texts = process_thread(thread)
        chunks = create_chunks(texts)
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{thread_id}_{i}"
            embedding = embeddings.embed_query(chunk.page_content)
            metadata = {
                "text": chunk.page_content,
                "thread_id": thread_id,
                "thread_title": thread['title'],
                "url": thread['url'],
                "timestamp": thread['scrape_time'],
                "chunk_index": i
            }
            update_document_in_index(index, chunk_id, embedding, metadata)
        
        st.session_state.processed_threads.add(thread_id)
        return len(chunks)
    except Exception as e:
        st.error(f"Errore processamento thread: {str(e)}")
        return 0

def render_database_cleanup(index):
    """Render database cleanup interface with proper deletion handling"""
    st.warning("⚠️ Danger Zone - Database Maintenance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧹 Clean Duplicates"):
            with st.spinner("Checking for duplicates..."):
                try:
                    # Get all documents
                    docs = fetch_all_documents(index)
                    if not docs:
                        st.info("No documents found in the database")
                        return
                    
                    # Create a map of content hashes to document IDs
                    content_map = {}
                    duplicates = []
                    
                    for doc in docs:
                        # Create a unique key based on thread and post IDs
                        content_key = f"{doc.metadata.get('thread_id')}_{doc.metadata.get('post_id')}"
                        
                        if content_key in content_map:
                            duplicates.append(doc.id)
                        else:
                            content_map[content_key] = doc.id
                    
                    if duplicates:
                        # Delete duplicates in batches
                        batch_size = 100
                        progress_bar = st.progress(0)
                        
                        for i in range(0, len(duplicates), batch_size):
                            batch = duplicates[i:i + batch_size]
                            try:
                                index.delete(ids=batch)
                                st.write(f"Deleted batch of {len(batch)} documents")
                            except Exception as e:
                                st.error(f"Error deleting batch: {str(e)}")
                            
                            progress = min(1.0, (i + batch_size) / len(duplicates))
                            progress_bar.progress(progress)
                        
                        st.success(f"Removed {len(duplicates)} duplicate documents")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.info("No duplicates found")
                        
                except Exception as e:
                    st.error(f"Error cleaning duplicates: {str(e)}")
    
    with col2:
        if st.button("🗑️ Clear Database", type="secondary"):
            try:
                # Delete all vectors
                index.delete(delete_all=True)
                st.success("Database cleared successfully!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing database: {str(e)}")
                st.error("Full error:", exception=e)

def fetch_all_documents(index):
    """Fetch all documents from index with proper error handling"""
    try:
        response = index.query(
            vector=[1.0] + [0.0] * 1535,  # Vector with first element non-zero
            top_k=10000,
            include_metadata=True
        )
        return response.matches if response and hasattr(response, 'matches') else []
    except Exception as e:
        st.error(f"Error fetching documents: {str(e)}")
        return []

def integrate_database_cleanup(index):
    """Integration point for the database cleanup functionality"""
    tabs = st.tabs(["📊 Overview", "🧹 Maintenance"])
    
    with tabs[0]:
        display_database_view(index)
    
    with tabs[1]:
        render_database_cleanup(index)

def verify_delete_permissions(index):
    """Verify delete permissions on the index"""
    try:
        # Create a test vector with some non-zero values
        test_id = "test_permissions"
        test_vector = [0.0] * 1536
        test_vector[0] = 1.0  # Set first value to 1.0
        test_vector[-1] = 0.5  # Set last value to 0.5
        
        # Insert test vector
        index.upsert(
            vectors=[{
                "id": test_id,
                "values": test_vector,
                "metadata": {"test": True}
            }]
        )
        
        # Small delay to ensure upsert is processed
        time.sleep(0.5)
        
        # Try to delete it
        index.delete(ids=[test_id])
        
        # Small delay to ensure delete is processed
        time.sleep(0.5)
        
        # Verify deletion
        verification = index.fetch(ids=[test_id])
        if verification and hasattr(verification, 'vectors') and verification.vectors:
            return False, "Insufficient delete permissions"
            
        return True, "Delete permissions verified"
        
    except Exception as e:
        return False, f"Error verifying permissions: {str(e)}"

def display_chat_interface(index, embeddings):
    """Display chat interface with improved styling."""
    # Check if database is empty
    stats = index.describe_index_stats()
    if stats['total_vector_count'] == 0:
        st.warning("Database is empty. Please load data from the Database tab.")
        return
    
    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Dimmi figliuolo..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        try:
            retriever = SmartRetriever(index, embeddings)
            chain = setup_rag_chain(retriever)
            
            with st.chat_message("assistant"):
                with st.spinner("Processing..."):
                    response = chain({"query": prompt})
                    st.markdown(response["result"])
            
            st.session_state.messages.append(
                {"role": "assistant", "content": response["result"]}
            )
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_database_view(index):
    """Visualizzazione e gestione del database."""
    st.header("📊 Database Management")
    
    # Statistiche generali
    try:
        stats = index.describe_index_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Documents", stats['total_vector_count'])
        with col2:
            st.metric("Dimension", stats['dimension'])
    except Exception as e:
        st.error(f"Error retrieving database stats: {str(e)}")
        return

    # Gestione documenti
    if st.button("📥 Load Documents", use_container_width=True):
        with st.spinner("Loading documents..."):
            try:
                # Create a zero vector with correct dimension (768)
                query_vector = [0.0] * 768
                query_vector[0] = 1.0  # Set first element to 1.0
                
                # Query with the correct dimension vector
                results = index.query(
                    vector=query_vector,
                    top_k=10000,
                    include_metadata=True
                )
                
                if not results.matches:
                    st.info("No documents found in the database")
                    return
                    
                # Process results as before
                data = []
                for doc in results.matches:
                    data.append({
                        'ID': doc.id,
                        'Thread': doc.metadata.get('thread_title', 'N/A'),
                        'Author': doc.metadata.get('author', 'N/A'),
                        'Date': doc.metadata.get('post_time', 'N/A'),
                        'URL': doc.metadata.get('url', 'N/A')
                    })
                
                df = pd.DataFrame(data)
                
                # Filtri
                st.subheader("🔍 Filters")
                col1, col2 = st.columns(2)
                with col1:
                    thread_filter = st.multiselect(
                        "Filter by Thread",
                        options=sorted(df['Thread'].unique())
                    )
                with col2:
                    author_filter = st.multiselect(
                        "Filter by Author",
                        options=sorted(df['Author'].unique())
                    )
                
                if thread_filter:
                    df = df[df['Thread'].isin(thread_filter)]
                if author_filter:
                    df = df[df['Author'].isin(author_filter)]
                
                st.subheader("📋 Documents")
                selected_rows = st.data_editor(
                    df,
                    hide_index=True,
                    use_container_width=True,
                    num_rows="dynamic"
                )
                
                if selected_rows is not None and len(selected_rows) > 0:
                    st.subheader("📝 Document Details")
                    selected_doc = next(doc for doc in results.matches if doc.id == selected_rows.iloc[0]['ID'])
                    if selected_doc:
                        with st.expander("Metadata", expanded=True):
                            st.json(selected_doc.metadata)
                            
            except Exception as e:
                st.error(f"Error fetching documents: {str(e)}")


def process_uploaded_file(uploaded_file, index, embeddings):
    """Process uploaded JSON file with button in sidebar."""
    if uploaded_file:
        if st.sidebar.button("Process File", key="process_file", use_container_width=True):
            with st.spinner("Processing file..."):
                data = load_json(uploaded_file)
                if data:
                    progress = st.progress(0)
                    total_chunks = 0
                    
                    for i, thread in enumerate(data):
                        chunks = process_and_index_thread(thread, embeddings, index)
                        total_chunks += chunks
                        progress.progress((i + 1) / len(data))
                    
                    st.success(f"Processed {len(data)} threads and created {total_chunks} chunks")


class RateLimiter:
    def __init__(self):
        self.requests = {}

    def __call__(self, func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            now = time.time()
            
            # Get or create request queue for this session
            session_id = id(st.session_state)
            if session_id not in self.requests:
                self.requests[session_id] = deque()
            
            # Clean old requests
            while (self.requests[session_id] and 
                   self.requests[session_id][0] < now - RATE_LIMIT_PERIOD):
                self.requests[session_id].popleft()
            
            # Check rate limit
            if len(self.requests[session_id]) >= MAX_REQUESTS_PER_MINUTE:
                wait_time = self.requests[session_id][0] + RATE_LIMIT_PERIOD - now
                if wait_time > 0:
                    st.warning(f"Rate limit exceeded. Please wait {int(wait_time)} seconds.")
                    time.sleep(min(wait_time, 5))  # Max wait of 5 seconds
            
            # Add current request
            self.requests[session_id].append(now)
            
            return func(*args, **kwargs)
        return wrapped

# Usage in display_chat_interface
rate_limiter = RateLimiter()

@rate_limiter
def process_chat_message(prompt, retriever, chain):
    response = chain({"query": prompt})
    return response["result"]


def main():
    # Apply custom styles
    apply_custom_styles()
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar and get selected option
    selected, uploaded_file = render_sidebar()
    
    try:
        index = initialize_pinecone()
        if index is None:
            st.stop()
        
        embeddings = get_embeddings()
        
        if uploaded_file:
            process_uploaded_file(uploaded_file, index, embeddings)
        
        if "Chat" in selected:
            st.markdown("## 💬 Chat")
            display_chat_interface(index, embeddings)
            
        elif "Database" in selected:
            display_database_view(index)
            
        else:  # Settings
            st.markdown("## ⚙️ Settings")
            render_database_cleanup(index)
            
    except Exception as e:
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()