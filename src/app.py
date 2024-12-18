import streamlit as st
from data.loader import load_json
from data.processor import process_thread
from embeddings.generator import create_chunks, get_embeddings
from embeddings.indexer import ensure_index_exists, update_document_in_index
from rag.retriever import PineconeRetriever
from rag.chain import setup_rag_chain
from ui.utils import display_thread_preview
import pinecone
import hashlib
from datetime import datetime

st.set_page_config(page_title="🔮 L'Oracolo", layout="wide")

def initialize_session_state():
    """Inizializza le variabili di sessione."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'processed_threads' not in st.session_state:
        st.session_state.processed_threads = set()

def get_thread_id(thread):
    """Genera un ID unico per il thread."""
    thread_key = f"{thread['url']}_{thread['scrape_time']}"
    return hashlib.md5(thread_key.encode()).hexdigest()

def process_and_index_thread(thread, embeddings, index):
    """Processa e indicizza un thread."""
    thread_id = get_thread_id(thread)
    
    # Controlla se il thread è già stato processato
    if thread_id in st.session_state.processed_threads:
        st.info(f"Thread '{thread['title']}' già processato. Verifico aggiornamenti...")
    
    # Processa i contenuti
    texts = process_thread(thread)
    chunks = create_chunks(texts)
    
    # Genera embeddings e aggiorna l'indice
    with st.spinner("Generazione embeddings e indicizzazione..."):
        for i, chunk in enumerate(chunks):
            chunk_id = f"{thread_id}_{i}"
            embedding = embeddings.embed_query(chunk.page_content)
            metadata = {
                "text": chunk.page_content,
                "thread_id": thread_id,
                "thread_title": thread['title'],
                "url": thread['url'],
                "timestamp": thread['scrape_time']
            }
            update_document_in_index(index, chunk_id, embedding, metadata)
    
    st.session_state.processed_threads.add(thread_id)
    return len(chunks)

def main():
    initialize_session_state()
    st.title("🔮 L'Oracolo")
    st.write("Un sistema RAG per analizzare le discussioni del forum")
    
    try:
        # Inizializza Pinecone
        pinecone.init(
            api_key=st.secrets["PINECONE_API_KEY"],
            environment=st.secrets["PINECONE_ENVIRONMENT"]
        )
        index = ensure_index_exists()
        embeddings = get_embeddings()
    except Exception as e:
        st.error(f"Errore di inizializzazione: {str(e)}")
        return
    
    # Sidebar per upload e processing
    with st.sidebar:
        st.header("Caricamento Dati")
        uploaded_file = st.file_uploader("Carica JSON del forum", type=['json'])
        
        if uploaded_file and st.button("Processa"):
            data = load_json(uploaded_file)
            if data:
                total_chunks = 0
                for thread in data:
                    st.write(f"Processamento thread: {thread['title']}")
                    chunks = process_and_index_thread(thread, embeddings, index)
                    total_chunks += chunks
                st.success(f"Processati {len(data)} thread e creati {total_chunks} chunks!")
                st.session_state['data'] = data
    
    # Mostra anteprima dati se disponibili
    if 'data' in st.session_state:
        with st.expander("📊 Anteprima Dati Caricati"):
            for thread in st.session_state['data']:
                display_thread_preview(thread)
    
    # Chat interface
    st.header("💬 Chat con l'Oracolo")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Chiedi all'Oracolo..."):
        if not st.session_state.processed_threads:
            st.warning("Per favore, carica e processa prima alcuni dati.")
            return
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        try:
            # Setup retriever e chain
            retriever = PineconeRetriever(index, embeddings)
            chain = setup_rag_chain(retriever)
            
            # Genera risposta
            with st.chat_message("assistant"):
                with st.spinner("Consulto la mia conoscenza..."):
                    response = chain({"query": prompt})
                    st.markdown(response['result'])
                    
                    # Mostra fonti
                    if st.toggle("Mostra fonti"):
                        st.divider()
                        for doc in response['source_documents']:
                            st.info(doc.metadata.get('thread_title'))
                            st.markdown(doc.page_content)
            
            st.session_state.messages.append({"role": "assistant", "content": response['result']})
            
        except Exception as e:
            st.error(f"Errore durante la generazione della risposta: {str(e)}")

if __name__ == "__main__":
    main()