import streamlit as st
from data.loader import load_json
from data.processor import process_thread
from embeddings.generator import create_chunks, get_embeddings
from embeddings.indexer import ensure_index_exists, update_document_in_index
from rag.retriever import PineconeRetriever
from rag.chain import setup_rag_chain
from ui.utils import display_thread_preview
import pinecone
from pinecone.core.client.configuration import Configuration as OpenApiConfiguration
import hashlib
from datetime import datetime
from config import INDEX_NAME
import logging

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

def get_thread_id(thread):
    """Genera un ID unico per il thread."""
    thread_key = f"{thread['url']}_{thread['scrape_time']}"
    return hashlib.md5(thread_key.encode()).hexdigest()

def reinit_pinecone():
    """Reinizializza la connessione a Pinecone con test DNS."""
    import socket
    import urllib3
    import ssl
    
    # Debug DNS
    st.write("Testing DNS resolution...")
    try:
        ip = socket.gethostbyname('controller.us-east-1.pinecone.io')
        st.write(f"DNS Resolution successful: {ip}")
    except socket.gaierror as e:
        st.write(f"DNS Resolution failed: {e}")
        st.write("Trying alternative DNS servers...")
        
        # Prova con i DNS di Google
        dns_servers = ['8.8.8.8', '8.8.4.4']
        for dns in dns_servers:
            try:
                resolver = dns.resolver.Resolver()
                resolver.nameservers = [dns]
                answer = resolver.resolve('controller.us-east-1.pinecone.io', 'A')
                st.write(f"Resolution with {dns} successful: {answer[0]}")
                ip = str(answer[0])
                break
            except Exception as e:
                st.write(f"Failed with {dns}: {e}")
    
    # Prova configurazione con pool personalizzato
    http = urllib3.PoolManager(
        timeout=urllib3.Timeout(connect=5, read=10),
        retries=urllib3.Retry(3),
        ssl_version=ssl.PROTOCOL_TLS
    )
    
    # Aggiungi gli header necessari
    headers = {
        'Api-Key': st.secrets["PINECONE_API_KEY"],
    }
    
    openapi_config = OpenApiConfiguration.get_default_copy()
    openapi_config.http_client = http
    openapi_config.ssl_ca_cert = None  # Disabilita la verifica del certificato per test
    openapi_config.assert_hostname = False
    
    try:
        pinecone.init(
            api_key=st.secrets["PINECONE_API_KEY"],
            environment=st.secrets["PINECONE_ENVIRONMENT"],
            openapi_config=openapi_config
        )
        st.write("Pinecone initialization successful")
    except Exception as e:
        st.write(f"Pinecone initialization failed: {str(e)}")
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
                "timestamp": thread['scrape_time']
            }
            # Reinizializza Pinecone prima di ogni operazione di update
            reinit_pinecone()
            update_document_in_index(index, chunk_id, embedding, metadata)
    
    st.session_state.processed_threads.add(thread_id)
    return len(chunks)

def main():
    initialize_session_state()
    st.title("🔮 L'Oracolo")
    st.write("Un sistema RAG per analizzare le discussioni del forum")
    
    try:
        # Debug info
        st.write("Debug connection info:")
        st.write(f"Environment: {st.secrets['PINECONE_ENVIRONMENT']}")
        st.write(f"API Key length: {len(st.secrets['PINECONE_API_KEY'])}")
        st.write(f"Index name: {INDEX_NAME}")
        
        # Inizializzazione Pinecone
        st.write("Tentativo di connessione a Pinecone:")
        
        # Forza reinizializzazione
        reinit_pinecone()
        
        # Lista degli indici disponibili
        indexes = pinecone.list_indexes()
        st.write("Available indexes:", indexes)
        
        # Ottieni l'indice
        index = pinecone.Index(INDEX_NAME)
        embeddings = get_embeddings()
        
        # Verifica la connessione
        stats = index.describe_index_stats()
        st.write("Index stats:", stats)
        
        st.success("Connessione a Pinecone stabilita con successo!")
        
        # Sidebar per il caricamento dati
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
        
        # Visualizza anteprima dati caricati
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
                # Reinizializza Pinecone prima della query
                reinit_pinecone()
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
    
    except Exception as e:
        st.error(f"Errore di inizializzazione: {str(e)}")
        # Debug aggiuntivo per l'errore
        st.write("Error details:")
        st.write(f"Error type: {type(e)}")
        st.write(f"Error message: {str(e)}")
        if hasattr(e, 'response'):
            st.write(f"Response status: {e.response.status_code}")
            st.write(f"Response headers: {e.response.headers}")
            st.write(f"Response body: {e.response.text}")

if __name__ == "__main__":
    main()