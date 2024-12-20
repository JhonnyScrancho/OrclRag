from embeddings.generator import get_embeddings
from embeddings.indexer import ensure_index_exists
from data.loader import load_json
import streamlit as st
import json
from data.processor import process_thread, update_thread_in_index
import logging

logger = logging.getLogger(__name__)

async def process_uploaded_file(uploaded_file):
    """Process the uploaded JSON file and update the index."""
    try:
        with st.spinner("Processing uploaded file..."):
            # Carica il JSON
            content = load_json(uploaded_file)
            if not content:
                st.error("Error loading JSON file")
                return
            
            # Verifica che il contenuto sia una lista
            if not isinstance(content, list):
                content = [content]  # Se è un singolo thread, convertilo in lista
            
            # Inizializza progress bar
            progress_bar = st.progress(0)
            progress_text = st.empty()
            total_threads = len(content)
            
            # Prepara embeddings e index
            embeddings = get_embeddings()
            index = ensure_index_exists()
            
            total_posts = 0
            for i, thread in enumerate(content):
                try:
                    # Processa il thread
                    posts_processed = await update_thread_in_index(index, thread, embeddings)
                    total_posts += posts_processed
                    
                    # Aggiorna progresso
                    progress = (i + 1) / total_threads
                    progress_bar.progress(progress)
                    progress_text.text(f"Processing thread {i + 1}/{total_threads}")
                    
                except Exception as thread_error:
                    st.error(f"Error processing thread {i + 1}: {str(thread_error)}")
                    continue
            
            st.success(f"Successfully processed {total_posts} posts from {total_threads} threads!")
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e)  # Mostra traceback dettagliato