import streamlit as st
import json
from data.processor import process_thread
import logging

logger = logging.getLogger(__name__)

def process_uploaded_file(uploaded_file):
    """Process and index uploaded JSON file"""
    try:
        with st.spinner("Processing file..."):
            if uploaded_file is not None:
                content = uploaded_file.read()
                data = json.loads(content)
                
                if not isinstance(data, list):
                    st.error("Invalid JSON format. Expected a list of threads.")
                    return False
                
                progress = st.progress(0)
                total_chunks = 0
                
                for i, thread in enumerate(data):
                    with st.status(f"Processing: {thread.get('title', 'Unknown thread')}", expanded=False) as status:
                        try:
                            chunks = process_thread(thread)
                            total_chunks += len(chunks)
                            status.update(label=f"Processed: {thread.get('title', 'Unknown thread')}", state="complete")
                        except Exception as e:
                            logger.error(f"Error processing thread: {str(e)}")
                            status.update(label=f"Error: {thread.get('title', 'Unknown thread')}", state="error")
                            continue
                        
                        progress.progress((i + 1) / len(data))
                
                st.success(f"Processed {len(data)} threads and created {total_chunks} chunks")
                st.session_state['data'] = data
                return True
    except json.JSONDecodeError:
        st.error("Invalid JSON file")
        return False
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return False
    
    return False