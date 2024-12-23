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
        from pinecone import Pinecone, ServerlessSpec
        st.write("Pinecone imported")
        
        # Initialize pinecone v3 style
        pc = Pinecone(
            api_key=st.secrets["PINECONE_API_KEY"]
        )
        st.write("Pinecone instance created")
        
        # Debug available indexes
        indexes = pc.list_indexes()
        st.write(f"Available indexes: {indexes}")
        
        # Get index
        index = pc.Index(INDEX_NAME)
        st.write("Got index")
        st.write(f"Index type: {type(index)}")
        st.write(f"Index methods: {dir(index)}")
        
        # Try to use stats() instead of describe_index_stats()
        try:
            stats = index.describe_index_stats()
            st.write("Stats method worked:", stats)
        except AttributeError:
            st.write("Trying alternative stats method...")
            try:
                # Try different methods that might exist
                if hasattr(index, 'stats'):
                    stats = index.stats()
                    st.write("Alternative stats worked:", stats)
                elif hasattr(index, 'describe'):
                    stats = index.describe()
                    st.write("Describe method worked:", stats)
                else:
                    st.write("No stats method found")
                    stats = {"dimension": 768}  # Default fallback
            except Exception as e:
                st.error(f"Alternative stats failed: {str(e)}")
                stats = {"dimension": 768}  # Default fallback
        
        return index
            
    except Exception as e:
        st.error(f"Pinecone initialization error: {str(e)}")
        st.error(f"Error type: {type(e)}")
        st.error(f"Error details: {dir(e) if hasattr(e, '__dict__') else 'No details available'}")
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
    """Fetch all documents from index with debug info."""
    try:
        st.write("Creating query vector...")
        # Create a normalized vector
        query_vector = [0.0] * 768
        query_vector[0] = 1.0
        
        st.write(f"Vector length: {len(query_vector)}")
        st.write(f"First few elements: {query_vector[:5]}")
        
        st.write("Sending query to Pinecone...")
        response = index.query(
            vector=query_vector,
            top_k=10000,
            include_metadata=True
        )
        
        return response.matches if response and hasattr(response, 'matches') else []
    except Exception as e:
        st.error(f"Error in fetch_all_documents: {str(e)}")
        raise

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
    print("DEBUG: Entering display_chat_interface")
    
    # Check if database is empty
    stats = index.describe_index_stats()
    print(f"DEBUG: Index stats in chat interface: {stats}")
    
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
    if prompt := st.chat_input("Chiedi all'Oracolo..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        try:
            print("DEBUG: Creating SmartRetriever...")
            retriever = SmartRetriever(index, embeddings)
            print("DEBUG: SmartRetriever created successfully")
            
            print("DEBUG: Setting up RAG chain...")
            chain = setup_rag_chain(retriever)
            print("DEBUG: RAG chain setup complete")
            
            with st.chat_message("assistant"):
                with st.spinner("Processing..."):
                    print("DEBUG: Processing query:", prompt)
                    response = chain({"query": prompt})
                    print("DEBUG: Got response from chain")
                    st.markdown(response["result"])
            
            st.session_state.messages.append(
                {"role": "assistant", "content": response["result"]}
            )
        except Exception as e:
            print(f"ERROR in chat processing: {str(e)}")
            st.error(f"Error generating response: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_database_view(index):
    """Visualizzazione e gestione del database."""
    st.header("📊 Database Management")
    
    # Statistiche generali
    try:
        stats = index.describe_index_stats()
        st.write("Index stats:", stats)  # Aggiungiamo questo
        
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
        try:
            st.write("Attempting to fetch documents...")
            # Usa il nuovo retriever con debug
            documents = fetch_all_documents(index)
            st.write(f"Retrieved {len(documents)} documents")
            
            if documents:
                # Display dei dati
                data = []
                for doc in documents:
                    st.write("Document metadata:", doc.metadata)  # Debug
                    data.append({
                        'ID': doc.id,
                        'Thread': doc.metadata.get('thread_title', 'N/A'),
                        'Author': doc.metadata.get('author', 'N/A'),
                        'Date': doc.metadata.get('post_time', 'N/A'),
                        'URL': doc.metadata.get('url', 'N/A')
                    })
                
                df = pd.DataFrame(data)
                st.dataframe(df)
            else:
                st.info("No documents found in the database")
                
        except Exception as e:
            st.error("Error in display_database_view:")
            st.error(f"Error type: {type(e)}")
            st.error(f"Error message: {str(e)}")

def process_uploaded_file(uploaded_file, index, embeddings):
    """Process uploaded JSON file."""
    if st.button("Process File", key="process_file"):
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

def main():
    print("\nDEBUG: Starting main application...")
    
    # Apply custom styles
    apply_custom_styles()
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar and get selected option
    print("DEBUG: Rendering sidebar...")
    selected, uploaded_file = render_sidebar()
    print(f"DEBUG: Selected option: {selected}")
    
    try:
        print("DEBUG: Initializing Pinecone...")
        index = initialize_pinecone()
        if index is None:
            print("ERROR: Failed to initialize Pinecone")
            st.stop()
        
        print("DEBUG: Getting embeddings...")
        embeddings = get_embeddings()
        print("DEBUG: Embeddings model initialized")
        
        if "Chat" in selected:
            print("DEBUG: Entering Chat interface")
            st.markdown("## 💬 Chat")
            display_chat_interface(index, embeddings)
            
        elif "Database" in selected:
            if uploaded_file:
                process_uploaded_file(uploaded_file, index, embeddings)
            display_database_view(index)
            
        else:  # Settings
            st.markdown("## ⚙️ Settings")
            render_database_cleanup(index)
            
    except Exception as e:
        print(f"ERROR in main: {str(e)}")
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()