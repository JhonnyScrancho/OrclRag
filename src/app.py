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
    Esegue hard delete dei documenti specificati da Pinecone
    """
    try:
        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]
            
        # Verifica che i documenti esistano prima dell'eliminazione
        existing_docs = index.fetch(ids=doc_ids)
        if not existing_docs.vectors:
            st.warning("Nessun documento trovato con gli ID specificati")
            return False
            
        # Esegue l'hard delete
        index.delete(
            ids=doc_ids,
            namespace=""  # namespace di default
        )
        
        # Verifica che i documenti siano stati effettivamente eliminati
        verification = index.fetch(ids=doc_ids)
        if verification.vectors:
            st.error("Eliminazione non riuscita - i documenti sono ancora presenti")
            return False
            
        return True
        
    except Exception as e:
        st.error(f"Errore durante l'eliminazione: {str(e)}")
        return False

def delete_all_documents(index):
    """
    Esegue hard delete di tutti i documenti nell'indice
    """
    try:
        # Forza il delete_all con conferma esplicita
        index.delete(
            delete_all=True,
            namespace=""  # namespace di default
        )
        
        # Verifica che l'indice sia effettivamente vuoto
        stats = index.describe_index_stats()
        if stats['total_vector_count'] > 0:
            st.error("Eliminazione totale non riuscita - documenti ancora presenti")
            return False
            
        st.session_state.processed_threads.clear()
        return True
        
    except Exception as e:
        st.error(f"Errore durante l'eliminazione totale: {str(e)}")
        return False

def verify_delete_permissions(index):
    """
    Verifica i permessi di eliminazione sull'indice
    """
    try:
        # Tenta di eliminare un documento di test
        test_id = "test_permissions"
        test_vector = [0.0] * 1536
        
        # Inserisce un vettore di test
        index.upsert(
            vectors=[{
                "id": test_id,
                "values": test_vector,
                "metadata": {"test": True}
            }]
        )
        
        # Prova ad eliminarlo
        index.delete(ids=[test_id])
        
        # Verifica l'eliminazione
        verification = index.fetch(ids=[test_id])
        if verification.vectors:
            return False, "Permessi di eliminazione insufficienti"
            
        return True, "Permessi di eliminazione OK"
        
    except Exception as e:
        return False, f"Errore verifica permessi: {str(e)}"

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
            
            # Sostituisci questa parte
            if st.button("Elimina selezionati"):
                if selected_rows is not None and len(selected_rows) > 0:
                    # Verifica permessi
                    has_permissions, message = verify_delete_permissions(index)
                    if has_permissions:
                        if delete_documents(index, selected_rows['ID'].tolist()):
                            st.success("Documenti eliminati correttamente")
                            st.rerun()
                    else:
                        st.error(f"Impossibile procedere: {message}")
            
            # E questa parte
            if st.button("Elimina tutto"):
                if st.checkbox("Conferma eliminazione totale"):
                    # Verifica permessi
                    has_permissions, message = verify_delete_permissions(index)
                    if has_permissions:
                        if delete_all_documents(index):
                            st.success("Database svuotato correttamente")
                            st.rerun()
                    else:
                        st.error(f"Impossibile procedere: {message}")
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