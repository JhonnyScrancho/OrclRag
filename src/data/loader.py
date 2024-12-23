import json
from typing import Dict, List
import streamlit as st
import logging
from .processor import process_thread

logger = logging.getLogger(__name__)

def load_json(uploaded_file) -> List[Dict]:
    """Carica e valida il file JSON."""
    try:
        # Leggi il contenuto del file
        content = uploaded_file.read()
        content_str = content.decode('utf-8')
        data = json.loads(content_str)
        
        if not isinstance(data, list):
            raise ValueError("Il JSON deve contenere una lista di thread")
            
        total_posts = 0
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each thread
        processed_data = []
        for i, thread in enumerate(data):
            try:
                # Validate thread structure
                required_fields = ['url', 'title', 'scrape_time', 'posts']
                if not all(field in thread for field in required_fields):
                    logger.warning(f"Thread {i} missing required fields")
                    continue
                    
                # Ensure posts is a list
                if not isinstance(thread['posts'], list):
                    logger.warning(f"Thread {i} 'posts' is not a list")
                    continue
                    
                # Count posts
                total_posts += len(thread['posts'])
                
                # Add thread to processed data
                processed_data.append(thread)
                
                # Update progress
                progress = (i + 1) / len(data)
                progress_bar.progress(progress)
                status_text.text(f"Processed {i+1}/{len(data)} threads. Total posts: {total_posts}")
                
            except Exception as e:
                logger.error(f"Error processing thread {i}: {str(e)}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        if processed_data:
            st.success(f"Successfully loaded {len(processed_data)} threads with {total_posts} total posts")
            return processed_data
        else:
            st.warning("No valid threads found in the file")
            return None
            
    except json.JSONDecodeError:
        st.error("File JSON non valido")
        logger.error("Invalid JSON file")
        return None
    except Exception as e:
        st.error(f"Errore nel caricamento del file: {str(e)}")
        logger.error(f"File loading error: {str(e)}")
        return None