import streamlit as st
from data.loader import load_json
from data.processor import process_thread
from embeddings.generator import create_chunks, get_embeddings
from embeddings.indexer import ensure_index_exists, update_document_in_index
from rag.retriever import PineconeRetriever
from rag.chain import setup_rag_chain
from ui.utils import display_thread_preview
import hashlib
from datetime import datetime
from pinecone import Pinecone
from config import INDEX_NAME
import logging
import pandas as pd

# Configurazione logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="🔮 L'Oracolo", layout="wide")

def initialize_session_state():
    """Inizializza le variabili di sessione."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'processed_threads' not in st.session_state:
        st.session_state.processed_threads = set()
    if 'db_view' not in st.session_state:
        st.session_state.db_view = 'threads'  # or 'posts'

def get_thread_id(thread):
    """Genera un ID unico per il thread."""
    thread_key = f"{thread['url']}_{thread['scrape_time']}"
    return hashlib.md5(thread_key.encode()).hexdigest()

def initialize_pinecone():
    """Inizializza la connessione a Pinecone e restituisce l'indice."""
    try:
        pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        index = pc.Index(INDEX_NAME)
        st.success("Connessione a Pinecone stabilita con successo!")
        return index
    except Exception as e:
        st.error(f"Errore nell'inizializzazione di Pinecone: {str(e)}")
        raise

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
                "timestamp": thread['scrape_time'],
                "chunk_index": i
            }
            update_document_in_index(index, chunk_id, embedding, metadata)
    
    st.session_state.processed_threads.add(thread_id)
    return len(chunks)

def fetch_all_documents(index):
    """Recupera tutti i documenti dall'indice."""
    try:
        response = index.query(
            vector=[0] * 1536,  # dimensione standard per OpenAI embeddings
            top_k=10000,  # adjust based on your needs
            include_metadata=True
        )
        return response.matches
    except Exception as e:
        st.error(f"Errore nel recupero dei documenti: {str(e)}")
        return []

def delete_document(index, doc_id):
    """Elimina un documento dall'indice."""
    try:
        index.delete(ids=[doc_id])
        st.success(f"Documento {doc_id} eliminato con successo!")
    except Exception as e:
        st.error(f"Errore nell'eliminazione del documento: {str(e)}")

def delete_all_documents(index):
    """Elimina tutti i documenti dall'indice."""
    try:
        index.delete(delete_all=True)
        st.success("Tutti i documenti sono stati eliminati con successo!")
        st.session_state.processed_threads.clear()
    except Exception as e:
        st.error(f"Errore nell'eliminazione dei documenti: {str(e)}")

def display_database_view(index):
    """Visualizza e gestisce i contenuti del database."""
    st.header("📊 Gestione Database")
    
    # Tabs per diverse viste
    tab1, tab2 = st.tabs(["Vista Generale", "Gestione Documenti"])
    
    with tab1:
        st.subheader("Statistiche Database")
        try:
            stats = index.describe_index_stats()
            st.write(f"Totale documenti: {stats['total_vector_count']}")
            st.write(f"Dimensione totale: {stats['dimension']} dimensioni")
        except Exception as e:
            st.error(f"Errore nel recupero delle statistiche: {str(e)}")
    
    with tab2:
        st.subheader("Gestione Documenti")
        if st.button("🔄 Aggiorna Lista"):
            documents = fetch_all_documents(index)
            
            if documents:
                # Crea DataFrame per visualizzazione
                data = []
                for doc in documents:
                    data.append({
                        'ID': doc.id,
                        'Thread': doc.metadata.get('thread_title', 'N/A'),
                        'URL': doc.metadata.get('url', 'N/A'),
                        'Timestamp': doc.metadata.get('timestamp', 'N/A'),
                        'Chunk Index': doc.metadata.get('chunk_index', 'N/A')
                    })
                
                df = pd.DataFrame(data)
                
                # Filtri
                col1, col2 = st.columns(2)
                with col1:
                    thread_filter = st.multiselect(
                        "Filtra per Thread",
                        options=df['Thread'].unique()
                    )
                with col2:
                    date_filter = st.date_input(
                        "Filtra per Data",
                        value=None
                    )
                
                # Applica filtri
                if thread_filter:
                    df = df[df['Thread'].isin(thread_filter)]
                if date_filter:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                    df = df[df['Timestamp'].dt.date == date_filter]
                
                # Visualizza DataFrame
                st.dataframe(df)
                
                # Azioni di massa
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("❌ Elimina Tutti i Documenti", type="secondary"):
                        if st.checkbox("Conferma eliminazione di tutti i documenti"):
                            delete_all_documents(index)
                            st.rerun()
                
                # Eliminazione singola
                st.subheader("Elimina Documento Specifico")
                doc_id = st.text_input("ID Documento da eliminare")
                if doc_id and st.button("Elimina Documento"):
                    delete_document(index, doc_id)
                    st.rerun()
            else:
                st.info("Nessun documento trovato nel database")

def main():
    initialize_session_state()
    st.title("🔮 L'Oracolo")
    
    try:
        # Inizializzazione Pinecone
        index = initialize_pinecone()
        embeddings = get_embeddings()
        
        # Tabs principali
        tab1, tab2, tab3 = st.tabs(["Chat", "Caricamento Dati", "Gestione DB"])
        
        with tab1:
            # Interface chat
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
                    retriever = PineconeRetriever(index, embeddings)
                    chain = setup_rag_chain(retriever)
                    
                    with st.chat_message("assistant"):
                        with st.spinner("Consulto la mia conoscenza..."):
                            response = chain({"query": prompt})
                            st.markdown(response['result'])
                            
                            if st.toggle("Mostra fonti"):
                                st.divider()
                                for doc in response['source_documents']:
                                    st.info(doc.metadata.get('thread_title'))
                                    st.markdown(doc.page_content)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response['result']})
                    
                except Exception as e:
                    st.error(f"Errore durante la generazione della risposta: {str(e)}")
        
        with tab2:
            # Caricamento dati
            st.header("📤 Caricamento Dati")
            uploaded_file = st.file_uploader("Carica JSON del forum", type=['json'])
            
            if uploaded_file and st.button("Processa", key="process_button"):
                data = load_json(uploaded_file)
                if data:
                    progress_bar = st.progress(0)
                    total_chunks = 0
                    for i, thread in enumerate(data):
                        st.write(f"Processamento thread: {thread['title']}")
                        chunks = process_and_index_thread(thread, embeddings, index)
                        total_chunks += chunks
                        progress_bar.progress((i + 1) / len(data))
                    st.success(f"Processati {len(data)} thread e creati {total_chunks} chunks!")
                    st.session_state['data'] = data
            
            # Visualizza anteprima dati caricati
            if 'data' in st.session_state:
                st.header("📊 Anteprima Dati Caricati")
                st.divider()
                for thread in st.session_state['data']:
                    display_thread_preview(thread)
                    st.divider()
        
        with tab3:
            # Gestione database
            display_database_view(index)
    
    except Exception as e:
        st.error(f"Errore di inizializzazione: {str(e)}")
        st.write("Dettagli errore:")
        st.write(f"Tipo errore: {type(e)}")
        st.write(f"Messaggio errore: {str(e)}")
        if hasattr(e, 'response'):
            st.write(f"Status risposta: {e.response.status_code}")
            st.write(f"Headers risposta: {e.response.headers}")
            st.write(f"Body risposta: {e.response.text}")

if __name__ == "__main__":
    main()