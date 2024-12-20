from ui.components import fetch_all_documents
import streamlit as st
from rag.retriever import SmartRetriever
from rag.chain import setup_rag_chain
from data.loader import load_json
from data.processor import process_thread
from embeddings.generator import create_chunks
import time
import pandas as pd

def render_chat_interface(index, embeddings):
    """Render chat interface"""
    st.header("💬 Chat")
    
    # Verifica che ci siano dati nel database
    stats = index.describe_index_stats()
    if stats['total_vector_count'] == 0:
        st.warning("Il database è vuoto. Carica dei dati dalla tab 'Database'.")
        return
    
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

def render_database_view(index, embeddings=None):
    """Enhanced database management interface"""
    st.header("📊 Database Management")
    
    tabs = st.tabs(["📝 Browse", "📤 Upload", "🗑️ Cleanup"])
    
    with tabs[0]:
        display_database_view(index)
    with tabs[1]:
        display_upload_interface(index, embeddings)
    with tabs[2]:
        display_cleanup_interface(index)

def display_upload_interface(index, embeddings):
    """Interfaccia di caricamento migliorata."""
    st.header("📤 Caricamento Dati")
    uploaded_file = st.file_uploader("Carica JSON del forum", type=['json'])
    
    if uploaded_file:
        try:
            if st.button("Processa", type="primary"):
                with st.spinner("Caricamento del file..."):
                    data = load_json(uploaded_file)
                    
                    if not data:
                        st.error("Errore: JSON non valido o vuoto")
                        return
                    
                    if not isinstance(data, list):
                        data = [data]  # Converti singolo thread in lista
                    
                    total_threads = len(data)
                    progress = st.progress(0)
                    status_text = st.empty()
                    total_chunks = 0
                    
                    for i, thread in enumerate(data):
                        try:
                            thread_id = get_thread_id(thread)
                            
                            # Controlla se il thread è già stato processato
                            if thread_id in st.session_state.processed_threads:
                                status_text.text(f"Thread già processato: {thread['title']}")
                                continue
                            
                            status_text.text(f"Processamento {i+1}/{total_threads}: {thread['title']}")
                            
                            # Processa il thread
                            texts = process_thread(thread)
                            chunks = create_chunks(texts)
                            
                            for j, chunk in enumerate(chunks):
                                chunk_id = f"{thread_id}_{j}"
                                embedding = embeddings.embed_query(chunk.page_content)
                                
                                metadata = {
                                    "text": chunk.page_content,
                                    "thread_id": thread_id,
                                    "thread_title": thread['title'],
                                    "url": thread['url'],
                                    "timestamp": thread['scrape_time'],
                                    "chunk_index": j,
                                    "total_chunks": len(chunks)
                                }
                                
                                # Usa upsert invece di update
                                index.upsert(
                                    vectors=[{
                                        "id": chunk_id,
                                        "values": embedding,
                                        "metadata": metadata
                                    }]
                                )
                                
                                total_chunks += 1
                            
                            st.session_state.processed_threads.add(thread_id)
                            progress.progress((i + 1) / total_threads)
                            
                        except Exception as thread_error:
                            st.error(f"Errore nel thread {i+1}: {str(thread_error)}")
                            continue
                    
                    st.success(f"Elaborazione completata! Processati {total_threads} thread e creati {total_chunks} chunks")
                    st.session_state['data'] = data  # Salva i dati per l'anteprima
                    
                    # Mostra anteprima
                    if 'data' in st.session_state:
                        st.header("Anteprima")
                        for thread in st.session_state['data']:
                            with st.expander(thread['title']):
                                st.write(f"URL: {thread['url']}")
                                st.write(f"Data: {thread['scrape_time']}")
                                for post in thread['posts'][:3]:
                                    st.markdown(f"""
                                    **Autore:** {post['author']}  
                                    **Data:** {post['post_time']}  
                                    **Contenuto:** {post['content'][:200]}...  
                                    **Keywords:** {', '.join(post['keywords'])}
                                    ---
                                    """)
        
        except Exception as e:
            st.error(f"Errore durante il caricamento: {str(e)}")
            st.exception(e)

def display_database_view(index):
    """Gestione database migliorata."""
    try:
        stats = index.describe_index_stats()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Totale Documenti", f"{stats['total_vector_count']:,}")
        with col2:
            st.metric("Dimensione", f"{stats['dimension']:,}d")
        with col3:
            size_mb = (stats['total_vector_count'] * 1536 * 4) / (1024 * 1024)
            st.metric("Dimensione DB", f"{size_mb:.2f} MB")
    except Exception as e:
        st.error(f"Errore recupero statistiche: {str(e)}")
    
    # Gestione documenti
    if st.button("Aggiorna lista", type="primary"):
        documents = fetch_all_documents(index)
        if documents:
            data = [{
                'ID': doc.id,
                'Thread': doc.metadata.get('thread_title', 'N/A'),
                'URL': doc.metadata.get('url', 'N/A'),
                'Data': doc.metadata.get('timestamp', 'N/A')
            } for doc in documents]
            
            df = pd.DataFrame(data)
            
            # Filtri
            col1, col2 = st.columns(2)
            with col1:
                thread_filter = st.multiselect("Filtra per Thread", options=sorted(df['Thread'].unique()))
            with col2:
                date_filter = st.multiselect("Filtra per Data", options=sorted(df['Data'].unique()))
            
            # Applica filtri
            if thread_filter:
                df = df[df['Thread'].isin(thread_filter)]
            if date_filter:
                df = df[df['Data'].isin(date_filter)]
            
            # Mostra dataframe con selezione
            selected = st.data_editor(
                df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "URL": st.column_config.LinkColumn("URL")
                }
            )
            
            # Azioni sui documenti selezionati
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("❌ Elimina Selezionati", type="primary"):
                    if len(selected) > 0:
                        try:
                            with st.spinner("Eliminazione in corso..."):
                                index.delete(
                                    ids=selected['ID'].tolist(),
                                    namespace=""
                                )
                                st.success(f"Eliminati {len(selected)} documenti")
                                time.sleep(1)
                                st.rerun()
                        except Exception as e:
                            st.error(f"Errore durante l'eliminazione: {str(e)}")
                            st.exception(e)
            
            with col2:
                if st.button("🗑️ Svuota Database", type="primary"):
                    if st.checkbox("⚠️ Conferma eliminazione TOTALE", key="confirm_total_delete"):
                        try:
                            with st.spinner("Eliminazione database in corso..."):
                                index.delete(
                                    delete_all=True,
                                    namespace=""
                                )
                                st.success("Database svuotato con successo!")
                                time.sleep(1)
                                st.rerun()
                        except Exception as e:
                            st.error(f"Errore durante l'eliminazione totale: {str(e)}")
                            st.exception(e)
        else:
            st.info("Nessun documento nel database")

def display_cleanup_interface(index):
    """Interfaccia pulizia database."""
    st.warning("⚠️ Danger Zone")
    
    if st.button("🔍 Analizza Duplicati"):
        try:
            documents = fetch_all_documents(index)
            if not documents:
                st.info("Nessun documento nel database")
                return
                
            # Analisi duplicati qui...
            st.info("Funzionalità in sviluppo")
            
        except Exception as e:
            st.error(f"Errore analisi duplicati: {str(e)}")