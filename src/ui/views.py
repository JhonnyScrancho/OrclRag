import streamlit as st
from rag.retriever import SmartRetriever
from rag.chain import setup_rag_chain
from ui.components import render_database_browser, render_database_search, render_database_cleanup

def render_chat_interface(index, embeddings):
    """Render chat interface"""
    st.header("💬 Chat")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    if prompt := st.chat_input("Chiedi all'Oracolo..."):
        handle_chat_input(prompt, index, embeddings)

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