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

# CSS essenziale
st.markdown("""
<style>
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

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
    
    # Verify delete permissions first
    can_delete, message = verify_delete_permissions(index)
    if not can_delete:
        st.error(f"Insufficient permissions: {message}")
        return
    
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
                            index.delete(ids=batch)
                            progress = min(1.0, (i + batch_size) / len(duplicates))
                            progress_bar.progress(progress)
                        
                        st.success(f"Removed {len(duplicates)} duplicate documents")
                        time.sleep(1)  # Allow time for deletion to propagate
                        st.rerun()
                    else:
                        st.info("No duplicates found")
                        
                except Exception as e:
                    st.error(f"Error cleaning duplicates: {str(e)}")
    
    with col2:
        if st.button("🗑️ Clear Database", type="secondary"):
            # Add a text input for confirmation
            confirm_text = st.text_input(
                "Type 'DELETE' to confirm database clearing",
                key="confirm_delete"
            )
            
            if confirm_text == "DELETE":
                try:
                    with st.spinner("Deleting all records..."):
                        # Get all document IDs
                        docs = fetch_all_documents(index)
                        if not docs:
                            st.info("Database is already empty")
                            return
                            
                        all_ids = [doc.id for doc in docs]
                        
                        # Delete in batches to handle large datasets
                        batch_size = 100
                        total_deleted = 0
                        
                        progress_bar = st.progress(0)
                        
                        for i in range(0, len(all_ids), batch_size):
                            batch = all_ids[i:i + batch_size]
                            index.delete(ids=batch)
                            total_deleted += len(batch)
                            progress = min(1.0, total_deleted / len(all_ids))
                            progress_bar.progress(progress)
                        
                        # Verify deletion
                        time.sleep(1)  # Allow time for deletion to propagate
                        remaining_docs = fetch_all_documents(index)
                        if not remaining_docs:
                            st.success("Database cleared successfully!")
                            st.rerun()
                        else:
                            st.warning(f"Some records ({len(remaining_docs)}) could not be deleted. Please try again.")
                            
                except Exception as e:
                    st.error(f"Error clearing database: {str(e)}")
            elif confirm_text:
                st.error("Please type 'DELETE' to confirm")

def fetch_all_documents(index):
    """Fetch all documents from index with proper error handling"""
    try:
        response = index.query(
            vector=[0] * 1536,
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
    """Interfaccia chat."""
    st.header("💬 Chat con l'Oracolo")
    
    # Verifica che ci siano dati nel database
    stats = index.describe_index_stats()
    if stats['total_vector_count'] == 0:
        st.warning("Il database è vuoto. Non ci sono dati da consultare.")
        return
    
    # Mostra la cronologia dei messaggi
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input chat
    if prompt := st.chat_input("Chiedi all'Oracolo..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        try:
            retriever = SmartRetriever(index, embeddings)
            chain = setup_rag_chain(retriever)
            
            with st.chat_message("assistant"):
                with st.spinner("Elaborazione..."):
                    response = chain({"query": prompt})
                    st.markdown(response["result"])
            
            st.session_state.messages.append({"role": "assistant", "content": response["result"]})
        except Exception as e:
            st.error(f"Errore generazione risposta: {str(e)}")

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
            documents = fetch_all_documents(index)
            if documents:
                # Preparazione dati per il DataFrame
                data = []
                for doc in documents:
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
                
                # Applica filtri
                if thread_filter:
                    df = df[df['Thread'].isin(thread_filter)]
                if author_filter:
                    df = df[df['Author'].isin(author_filter)]
                
                # Visualizzazione dati
                st.subheader("📋 Documents")
                selected_rows = st.data_editor(
                    df,
                    hide_index=True,
                    use_container_width=True,
                    num_rows="dynamic"
                )
                
                # Mostra metadati dettagliati per la riga selezionata
                if selected_rows is not None and len(selected_rows) > 0:
                    st.subheader("📝 Document Details")
                    selected_doc = next(doc for doc in documents if doc.id == selected_rows.iloc[0]['ID'])
                    if selected_doc:
                        with st.expander("Metadata", expanded=True):
                            st.json(selected_doc.metadata)
            else:
                st.info("No documents found in the database")

def main():
    initialize_session_state()
    st.title("🔮 L'Oracolo")
    
    try:
        index = initialize_pinecone()
        if index is None:
            st.stop()
        
        embeddings = get_embeddings()
        
        # Mostra la chat come tab principale
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "📤 Caricamento", "🗄️ Database"])
        
        with tab1:
            display_chat_interface(index, embeddings)
        
        with tab2:
            st.header("📤 Caricamento Dati")
            uploaded_file = st.file_uploader("Carica JSON del forum", type=['json'])
            
            if uploaded_file and st.button("Processa"):
                data = load_json(uploaded_file)
                if data:
                    progress = st.progress(0)
                    total_chunks = 0
                    
                    for i, thread in enumerate(data):
                        st.text(f"Processamento: {thread['title']}")
                        chunks = process_and_index_thread(thread, embeddings, index)
                        total_chunks += chunks
                        progress.progress((i + 1) / len(data))
                    
                    st.success(f"Processati {len(data)} thread e creati {total_chunks} chunks")
                    st.session_state['data'] = data
            
            if 'data' in st.session_state:
                st.header("Anteprima")
                for thread in st.session_state['data']:
                    with st.expander(thread['title']):
                        st.write(f"URL: {thread['url']}")
                        st.write(f"Data: {thread['scrape_time']}")
                        for post in thread['posts']:
                            st.markdown(f"""
                            **Autore:** {post['author']}  
                            **Data:** {post['post_time']}  
                            **Contenuto:** {post['content']}  
                            **Keywords:** {', '.join(post['keywords'])}
                            ---
                            """)
        
        with tab3:
            integrate_database_cleanup(index)
    
    except Exception as e:
        st.error(f"Errore applicazione: {str(e)}")

if __name__ == "__main__":
    main()