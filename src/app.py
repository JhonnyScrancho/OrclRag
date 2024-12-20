import streamlit as st
from config import INDEX_NAME, LLM_MODEL
import streamlit.components.v1 as components
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
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config with custom theme
st.set_page_config(
    page_title="🔮 L'Oracolo",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Logo and title styling */
    img {
        border-radius: 60% !important;
    }
    
    .logo-title {
        text-align: center;
        font-size: 2em !important;
        font-weight: bold !important;
    }
    
    /* Modern color scheme */
    :root {
        --primary-color: #7C3AED;
        --secondary-color: #4F46E5;
        --background-color: #F9FAFB;
        --text-color: #1F2937;
        --sidebar-color: #F3F4F6;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--sidebar-color);
        padding: 2rem 1rem;
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Chat container */
    .chat-container {
        height: calc(100vh - 280px);
        overflow-y: auto;
        padding: 1rem;
        margin-bottom: 80px;
        background: white;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Fixed chat input at bottom */
    .stChatInputContainer {
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        max-width: 800px;
        background: white;
        padding: 1rem;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        z-index: 1000;
    }
    
    /* Modern button styling */
    .stButton>button {
        background: var(--primary-color);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        background: var(--secondary-color);
        transform: translateY(-1px);
    }
    
    /* Card styling */
    .card {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Message bubbles */
    .message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        max-width: 80%;
    }
    
    .user-message {
        background: var(--primary-color);
        color: white;
        margin-left: auto;
    }
    
    .assistant-message {
        background: var(--sidebar-color);
        color: var(--text-color);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: nowrap;
        border-radius: 6px;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .chat-container {
            height: calc(100vh - 200px);
        }
        
        .message {
            max-width: 90%;
        }
    }
</style>
""", unsafe_allow_html=True)

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

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'processed_threads' not in st.session_state:
        st.session_state.processed_threads = set()
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "chat"

def render_sidebar():
    """Enhanced sidebar with navigation and tools"""
    with st.sidebar:
        # Logo con bordo circolare
        st.image("src/img/logo.png", use_column_width=True)
        
        # Titolo sotto il logo
        st.markdown('<h1 class="logo-title">L\'Oracolo</h1>', unsafe_allow_html=True)
        
        # Main navigation
        selected = st.radio(
            "",
            ["💬 Chat", "📊 Database", "⚙️ Settings"],
            key="navigation"
        )
        
        st.session_state.current_page = selected.split()[1].lower()
        
        # Additional tools in sidebar
        st.divider()
        
        # File uploader in sidebar
        uploaded_file = st.file_uploader(
            "Upload Forum JSON",
            type=['json'],
            help="Upload a JSON file containing forum data"
        )
        
        if uploaded_file:
            if st.button("Process Data", type="primary"):
                process_uploaded_file(uploaded_file)

def render_chat_interface(index, embeddings):
    """Interfaccia chat pulita e semplificata"""
    st.header("💬 Chat")
    
    # Visualizza i messaggi
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Input chat in fondo
    if prompt := st.chat_input("Chiedi all'Oracolo..."):
        handle_chat_input(prompt, index, embeddings)

def render_database_view(index):
    """Enhanced database management interface"""
    st.header("📊 Database Management")
    
    # Database stats card
    with st.container():
        col1, col2, col3 = st.columns(3)
        try:
            stats = index.describe_index_stats()
            with col1:
                st.metric("Total Documents", f"{stats['total_vector_count']:,}")
            with col2:
                st.metric("Dimension", stats.get('dimension', 'N/A'))
            with col3:
                st.metric("Index Size", f"{stats.get('total_vector_count', 0) * 1536 * 4 / (1024*1024):.2f} MB")
        except Exception as e:
            st.error(f"Error fetching stats: {str(e)}")
    
    # Database operations
    st.subheader("Database Operations")
    tabs = st.tabs(["📝 Browse", "🔍 Search", "🗑️ Cleanup"])
    
    with tabs[0]:
        render_database_browser(index)
    with tabs[1]:
        render_database_search(index)
    with tabs[2]:
        render_database_cleanup(index)

def render_settings():
    """Settings page with configuration options"""
    st.header("⚙️ Settings")
    
    with st.expander("API Configuration"):
        st.text_input("OpenAI API Key", type="password", value="****", disabled=True)
        st.text_input("Pinecone API Key", type="password", value="****", disabled=True)
    
    with st.expander("Model Settings"):
        st.selectbox("LLM Model", ["gpt-4-turbo-preview", "gpt-3.5-turbo"])
        st.slider("Temperature", 0.0, 1.0, 0.3)
        st.number_input("Max Tokens", 100, 2000, 500)

def handle_chat_input(prompt: str, index, embeddings):
    """Process chat input and generate response"""
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    try:
        retriever = SmartRetriever(index, embeddings)
        chain = setup_rag_chain(retriever)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chain({"query": prompt})
                st.markdown(response["result"])
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["result"]
        })
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")

def process_uploaded_file(uploaded_file):
    """Process and index uploaded JSON file"""
    with st.spinner("Processing file..."):
        data = load_json(uploaded_file)
        if data:
            progress = st.progress(0)
            total_chunks = 0
            
            for i, thread in enumerate(data):
                # Usa st.status per mostrare il thread corrente
                with st.status(f"Processing: {thread['title']}", expanded=False) as status:
                    try:
                        # Process thread
                        chunks = process_thread(thread)
                        total_chunks += len(chunks)
                        status.update(label=f"Processed: {thread['title']}", state="complete")
                    except Exception as e:
                        logger.error(f"Error processing thread {thread.get('title', 'Unknown')}: {str(e)}")
                        status.update(label=f"Error: {thread['title']}", state="error")
                        continue
                    
                # Aggiorna la barra di progresso
                progress.progress((i + 1) / len(data))
            
            # Mostra il risultato finale
            st.success(f"Processed {len(data)} threads and created {total_chunks} chunks")
            st.session_state['data'] = data
            return True
    return False
    
def validate_thread(thread):
    """Validate thread structure before processing"""
    required_fields = ['title', 'url', 'posts']
    
    if not all(field in thread for field in required_fields):
        missing_fields = [field for field in required_fields if field not in thread]
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
    
    if not isinstance(thread['posts'], list):
        raise ValueError("'posts' must be a list")
    
    return True    

def render_database_browser(index):
    """Render database content browser"""
    if st.button("Refresh List"):
        documents = fetch_all_documents(index)
        if documents:
            df = pd.DataFrame([{
                'ID': doc.id,
                'Thread': doc.metadata.get('thread_title', 'N/A'),
                'URL': doc.metadata.get('url', 'N/A'),
                'Date': doc.metadata.get('timestamp', 'N/A'),
                'Author': doc.metadata.get('author', 'N/A')
            } for doc in documents])
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                thread_filter = st.multiselect(
                    "Filter by Thread",
                    options=df['Thread'].unique()
                )
            with col2:
                author_filter = st.multiselect(
                    "Filter by Author",
                    options=df['Author'].unique()
                )
            
            # Apply filters
            if thread_filter:
                df = df[df['Thread'].isin(thread_filter)]
            if author_filter:
                df = df[df['Author'].isin(author_filter)]
            
            # Display data table
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'URL': st.column_config.LinkColumn('URL'),
                    'Date': st.column_config.DateColumn('Date'),
                }
            )
        else:
            st.info("No documents found in database")

def render_database_search(index):
    """Render database search interface"""
    search_query = st.text_input("Search documents", placeholder="Enter search terms...")
    
    if search_query:
        with st.spinner("Searching..."):
            try:
                embeddings = get_embeddings()
                retriever = SmartRetriever(index, embeddings)
                docs = retriever.query_with_limit(search_query, limit=5)
                
                for doc in docs:
                    with st.expander(f"{doc.metadata.get('thread_title', 'Unknown Thread')}"):
                        st.markdown(f"""
                        **Author:** {doc.metadata.get('author', 'Unknown')}  
                        **Date:** {doc.metadata.get('post_time', 'Unknown')}  
                        **Content:**  
                        {doc.page_content}
                        """)
            except Exception as e:
                st.error(f"Search error: {str(e)}")

def generate_content_hash(metadata):
    """Generate a unique hash based on post content and relevant metadata"""
    hash_data = {
        'thread_id': metadata.get('thread_id'),
        'post_id': metadata.get('post_id'),
        'author': metadata.get('author'),
        'post_time': metadata.get('post_time'),
        'text': metadata.get('text'),
        'thread_title': metadata.get('thread_title')
    }
    hash_string = json.dumps(hash_data, sort_keys=True)
    return hashlib.md5(hash_string.encode()).hexdigest()

def analyze_duplicates(index):
    """Analyze and display duplicate content in the database"""
    st.subheader("🔍 Duplicate Analysis")
    
    try:
        with st.spinner("Analyzing database for duplicates..."):
            docs = fetch_all_documents(index)
            if not docs:
                st.info("No documents found in database")
                return
            
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            content_map = {}
            total_docs = len(docs)
            
            for i, doc in enumerate(docs):
                content_hash = generate_content_hash(doc.metadata)
                
                if content_hash not in content_map:
                    content_map[content_hash] = []
                content_map[content_hash].append({
                    'id': doc.id,
                    'thread_title': doc.metadata.get('thread_title', 'Unknown'),
                    'thread_id': doc.metadata.get('thread_id', 'Unknown'),
                    'post_id': doc.metadata.get('post_id', 'Unknown'),
                    'author': doc.metadata.get('author', 'Unknown'),
                    'post_time': doc.metadata.get('post_time', 'Unknown'),
                    'content_preview': doc.metadata.get('text', '')[:100] + '...' if doc.metadata.get('text') else 'No content'
                })
                
                progress = (i + 1) / total_docs
                progress_bar.progress(progress)
                progress_text.text(f"Analyzing documents: {i + 1}/{total_docs}")
            
            duplicates = {k: v for k, v in content_map.items() if len(v) > 1}
            
            if duplicates:
                st.warning(f"Found {len(duplicates)} groups of duplicate content")
                
                total_duplicates = sum(len(group) - 1 for group in duplicates.values())
                duplicate_percentage = (total_duplicates / total_docs) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Documents", total_docs)
                with col2:
                    st.metric("Duplicate Groups", len(duplicates))
                with col3:
                    st.metric("Duplicate Percentage", f"{duplicate_percentage:.1f}%")
                
                duplicate_records = []
                for content_hash, docs in duplicates.items():
                    for doc in docs:
                        duplicate_records.append({
                            'Hash': content_hash[:8] + '...',
                            'Thread Title': doc['thread_title'],
                            'Author': doc['author'],
                            'Post Time': doc['post_time'],
                            'Document ID': doc['id'],
                            'Content Preview': doc['content_preview']
                        })
                
                df = pd.DataFrame(duplicate_records)
                
                st.subheader("Duplicate Details")
                col1, col2 = st.columns(2)
                with col1:
                    thread_filter = st.multiselect(
                        "Filter by Thread",
                        options=sorted(df['Thread Title'].unique())
                    )
                with col2:
                    author_filter = st.multiselect(
                        "Filter by Author",
                        options=sorted(df['Author'].unique())
                    )
                
                if thread_filter:
                    df = df[df['Thread Title'].isin(thread_filter)]
                if author_filter:
                    df = df[df['Author'].isin(author_filter)]
                
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )
                
                if st.button("Export Duplicate Analysis"):
                    st.download_button(
                        label="Download CSV",
                        data=df.to_csv(index=False).encode('utf-8'),
                        file_name="duplicate_analysis.csv",
                        mime="text/csv"
                    )
            else:
                st.success("No duplicates found in the database!")
                
    except Exception as e:
        st.error(f"Error analyzing duplicates: {str(e)}")
        raise

def render_database_cleanup(index):
    """Render database cleanup interface"""
    st.warning("⚠️ Danger Zone")
    
    # Analyze duplicates
    analyze_duplicates(index)
    
    # Clear database section
    st.divider()
    if st.button("🗑️ Clear Database", type="secondary"):
        confirm_text = st.text_input(
            "Type 'DELETE' to confirm database clearing",
            key="confirm_delete"
        )
        
        if confirm_text == "DELETE":
            try:
                with st.spinner("Deleting all records..."):
                    docs = fetch_all_documents(index)
                    all_ids = [doc.id for doc in docs]
                    
                    batch_size = 100
                    total_batches = (len(all_ids) + batch_size - 1) // batch_size
                    
                    progress_bar = st.progress(0)
                    progress_text = st.empty()
                    
                    for batch_num, i in enumerate(range(0, len(all_ids), batch_size)):
                        batch = all_ids[i:i + batch_size]
                        index.delete(ids=batch)
                        
                        progress = (batch_num + 1) / total_batches
                        progress_bar.progress(progress)
                        progress_text.text(f"Deleting batch {batch_num + 1}/{total_batches}")
                    
                    # Verify deletion
                    remaining_docs = fetch_all_documents(index)
                    if not remaining_docs:
                        st.success("Database cleared successfully!")
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
        return response.matches if response else []
    except Exception as e:
        st.error(f"Error fetching documents: {str(e)}")
        return []

def main():
    """Main application with improved navigation and UI"""
    initialize_session_state()
    
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