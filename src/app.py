import streamlit as st
from config import INDEX_NAME, LLM_MODEL 
from data.loader import load_json
from data.processor import process_thread
from embeddings.generator import create_chunks, get_embeddings
from embeddings.indexer import update_document_in_index
from rag.retriever import PineconeRetriever
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

def fetch_all_documents(index):
    """Recupera tutti i documenti dall'indice."""
    try:
        response = index.query(
            vector=[0] * 1536,
            top_k=10000,
            include_metadata=True
        )
        return response.matches
    except Exception as e:
        st.error(f"Errore recupero documenti: {str(e)}")
        return []

def delete_documents(index, doc_ids):
    """
    Elimina documenti da Pinecone in batch per rispettare i rate limits
    """
    try:
        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]
        
        # Dividi in batch di massimo 100 ID per richiesta
        BATCH_SIZE = 100
        for i in range(0, len(doc_ids), BATCH_SIZE):
            batch = doc_ids[i:i + BATCH_SIZE]
            
            # Aggiungi un piccolo delay tra le richieste
            time.sleep(0.5)
            
            try:
                index.delete(ids=batch)
                st.write(f"Eliminato batch {i//BATCH_SIZE + 1}/{(len(doc_ids) + BATCH_SIZE - 1)//BATCH_SIZE}")
            except Exception as batch_error:
                st.warning(f"Errore nell'eliminazione del batch {i//BATCH_SIZE + 1}: {str(batch_error)}")
                continue
        
        return True
    except Exception as e:
        st.error(f"Errore eliminazione documenti: {str(e)}")
        return False

def delete_all_documents(index):
    """
    Elimina tutti i documenti rispettando i rate limits
    """
    try:
        # Prima recupera tutti gli ID
        results = index.query(
            vector=[0] * 1536,
            top_k=10000,
            include_metadata=True
        )
        
        if not results.matches:
            st.warning("Nessun documento da eliminare")
            return True
            
        # Estrai gli ID
        doc_ids = [match.id for match in results.matches]
        
        # Usa la funzione batch per eliminare
        success = delete_documents(index, doc_ids)
        
        if success:
            st.session_state.processed_threads.clear()
            return True
            
        return False
        
    except Exception as e:
        st.error(f"Errore eliminazione totale: {str(e)}")
        return False

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
            retriever = PineconeRetriever(index, embeddings)
            chain = setup_rag_chain(retriever)
            
            with st.chat_message("assistant"):
                with st.spinner("Elaborazione..."):
                    response = chain({"query": prompt})
                    st.markdown(response["result"])
                    
                    if st.toggle("Mostra fonti"):
                        for doc in retriever.get_relevant_documents(prompt):
                            st.info(doc.metadata.get('thread_title'))
                            st.markdown(doc.page_content)
            
            st.session_state.messages.append({"role": "assistant", "content": response["result"]})
        except Exception as e:
            st.error(f"Errore generazione risposta: {str(e)}")

def display_database_view(index):
    """Gestione database."""
    st.header("📊 Database")
    
    # Statistiche
    try:
        stats = index.describe_index_stats()
        st.info(f"Documenti nel database: {stats['total_vector_count']}")
    except Exception as e:
        st.error(f"Errore recupero statistiche: {str(e)}")
    
    # Gestione documenti
    if st.button("Aggiorna lista"):
        documents = fetch_all_documents(index)
        if documents:
            data = [{
                'ID': doc.id,
                'Thread': doc.metadata.get('thread_title', 'N/A'),
                'URL': doc.metadata.get('url', 'N/A'),
                'Data': doc.metadata.get('timestamp', 'N/A')
            } for doc in documents]
            
            df = pd.DataFrame(data)
            
            # Filtri base
            thread_filter = st.multiselect("Filtra per Thread", options=df['Thread'].unique())
            if thread_filter:
                df = df[df['Thread'].isin(thread_filter)]
            
            selected_rows = st.data_editor(
                df,
                hide_index=True,
                use_container_width=True
            )
            
            if st.button("Elimina selezionati"):
                if selected_rows is not None and len(selected_rows) > 0:
                    if delete_documents(index, selected_rows['ID'].tolist()):
                        st.success("Documenti eliminati")
                        st.rerun()
            
            if st.button("Elimina tutto"):
                if st.checkbox("Conferma eliminazione totale"):
                    if delete_all_documents(index):
                        st.success("Database svuotato")
                        st.rerun()
        else:
            st.info("Nessun documento nel database")

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
            display_database_view(index)
    
    except Exception as e:
        st.error(f"Errore applicazione: {str(e)}")

if __name__ == "__main__":
    main()