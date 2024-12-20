from rag.retriever import SmartRetriever
import streamlit as st
import pandas as pd
import json
import hashlib
from data.loader import load_json
from data.processor import process_thread
from core.processor import process_uploaded_file

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
                from embeddings.generator import get_embeddings
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
                content_hash = hashlib.md5(
                    json.dumps(doc.metadata, sort_keys=True).encode()
                ).hexdigest()
                
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