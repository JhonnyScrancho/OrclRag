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
        # Get index stats to get the correct dimension
        stats = index.describe_index_stats()
        dimension = stats['dimension']
        
        # Create query vector with correct dimension
        query_vector = [0.0] * dimension
        query_vector[0] = 1.0  # Set first element to 1.0
        
        response = index.query(
            vector=query_vector,
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
        avatar = "🫏" if message["role"] == "user" else "🧚"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Dimmi figliuolo..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🫏"):
            st.markdown(prompt)
        
        try:
            retriever = SmartRetriever(index, embeddings)
            chain = setup_rag_chain(retriever)
            
            with st.chat_message("assistant", avatar="🧚"):
                with st.spinner("Sto creando..."):
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
    
    # Initialize session states
    if 'selected_thread' not in st.session_state:
        st.session_state.selected_thread = None
    if 'threads_data' not in st.session_state:
        st.session_state.threads_data = None
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = None
    
    try:
        stats = index.describe_index_stats()
        
        # Calcola il numero di thread unici
        results = index.query(
            vector=[1.0] + [0.0] * (stats['dimension'] - 1),
            top_k=10000,
            include_metadata=True
        )
        
        unique_threads = len(set(doc.metadata.get('thread_id', '') for doc in results.matches))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Documents", stats['total_vector_count'])
        with col2:
            st.metric("Total Threads", unique_threads)
            
    except Exception as e:
        st.error(f"Error retrieving database stats: {str(e)}")
        return

    if st.button("📥 Load Documents", use_container_width=True) or st.session_state.threads_data is not None:
        if st.session_state.threads_data is None:  # Solo se i dati non sono già caricati
            with st.spinner("Loading documents..."):
                try:
                    dimension = stats['dimension']
                    query_vector = [0.0] * dimension
                    query_vector[0] = 1.0
                    
                    results = index.query(
                        vector=query_vector,
                        top_k=10000,
                        include_metadata=True
                    )
                    
                    if not results.matches:
                        st.info("No documents found in the database")
                        return
                    
                    # Processa i risultati
                    threads_data = {}
                    for doc in results.matches:
                        thread_id = doc.metadata.get('thread_id')
                        if thread_id not in threads_data:
                            threads_data[thread_id] = {
                                'Thread ID': thread_id,
                                'Title': doc.metadata.get('thread_title'),
                                'URL': doc.metadata.get('url'),
                                'Posts': [],
                            }
                        
                        text = doc.metadata.get('text', '')
                        post_data = parse_post_content(text)
                        
                        if post_data:
                            threads_data[thread_id]['Posts'].append(post_data)
                    
                    # Calcola totali
                    for thread_id in threads_data:
                        threads_data[thread_id]['Total Posts'] = len(threads_data[thread_id]['Posts'])
                    
                    # Salva i dati nello state
                    st.session_state.threads_data = threads_data
                    
                    # Crea DataFrame
                    threads_list = []
                    for thread_data in threads_data.values():
                        threads_list.append({
                            'Thread ID': thread_data['Thread ID'],
                            'Title': thread_data['Title'],
                            'URL': thread_data['URL'],
                            'Total Posts': thread_data['Total Posts']
                        })
                    
                    st.session_state.filtered_df = pd.DataFrame(threads_list)
                    
                except Exception as e:
                    st.error(f"Error fetching documents: {str(e)}")
                    st.exception(e)
                    return
        
        # Mostra i filtri e la lista solo se abbiamo dati
        if st.session_state.filtered_df is not None:
            # Filtri
            st.subheader("🔍 Filters")
            col1, col2 = st.columns(2)
            with col1:
                title_filter = st.multiselect(
                    "Filter by Title",
                    options=sorted(st.session_state.filtered_df['Title'].unique())
                )
            
            # Applica filtri
            filtered_df = st.session_state.filtered_df.copy()
            if title_filter:
                filtered_df = filtered_df[filtered_df['Title'].isin(title_filter)]
            
            # Lista thread
            st.subheader("📋 Thread List")
            
            # Visualizza thread con expander
            for index, row in filtered_df.iterrows():
                thread_id = row['Thread ID']
                thread_data = st.session_state.threads_data[thread_id]
                
                with st.expander(
                    f"🧵 {thread_data['Title']} ({thread_data['Total Posts']} posts)", 
                    expanded=(st.session_state.selected_thread == thread_id)
                ):
                    st.markdown(f"🔗 [Thread URL]({thread_data['URL']})")
                    
                    if st.session_state.selected_thread == thread_id:
                        st.button("Hide Posts", key=f"hide_{thread_id}", 
                                 on_click=lambda: setattr(st.session_state, 'selected_thread', None))
                        
                        for post in thread_data['Posts']:
                            st.markdown(f"""
                            **Author:** {post['author']}  
                            **Time:** {post['time']}  
                            
                            {format_post_content(post)}
                            ---
                            """)
                    else:
                        st.button("Load Posts", key=f"load_{thread_id}", 
                                 on_click=lambda tid=thread_id: setattr(st.session_state, 'selected_thread', tid))

def parse_post_content(text):
    """Estrae i metadati dal testo del post."""
    lines = text.strip().split('\n')
    post_data = {
        'author': '',
        'time': '',
        'content': '',
        'quoted_author': '',
        'quoted_content': ''
    }
    
    current_section = None
    content_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('Author:'):
            post_data['author'] = line.replace('Author:', '').strip()
        elif line.startswith('Time:'):
            post_data['time'] = line.replace('Time:', '').strip()
        elif line.startswith('Quoted Author:'):
            post_data['quoted_author'] = line.replace('Quoted Author:', '').strip()
        elif line.startswith('Quoted Content:'):
            post_data['quoted_content'] = line.replace('Quoted Content:', '').strip()
        elif line.startswith('Content:'):
            current_section = 'content'
        elif current_section == 'content':
            if not line.startswith('Keywords:') and not line.startswith('Sentiment:'):
                content_lines.append(line)
    
    post_data['content'] = ' '.join(content_lines).strip()
    return post_data

def format_post_content(post):
    """Formatta il contenuto del post con citazioni se presenti."""
    formatted_content = ""
    
    if post['quoted_author'] and post['quoted_content']:
        formatted_content += f"> **{post['quoted_author']} wrote:**  \n> {post['quoted_content']}\n\n"
    
    formatted_content += post['content']
    return formatted_content


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