import concurrent.futures
from typing import List, Dict, Any
from math import ceil
import streamlit as st
from config import (
    BATCH_SIZE, 
    MAX_TOKENS_PER_REQUEST, 
    THREADING_ENABLED,
    PARALLEL_PROCESSING_THREADS,
    POSTS_PER_PAGE
)

class BatchDocumentProcessor:
    def __init__(self, retriever, max_docs=None):
        self.retriever = retriever
        self.max_docs = max_docs
        self.processed_docs = 0
        self.total_threads = set()
        self.progress_bar = None

    def process_in_batches(self, documents: List[Any]) -> List[Dict]:
        """Process documents in batches with progress tracking."""
        if not documents:
            return []

        total_docs = len(documents)
        num_batches = ceil(total_docs / BATCH_SIZE)
        results = []
        
        # Initialize progress tracking
        if st.sidebar.checkbox("Show Processing Progress", value=True):
            self.progress_bar = st.progress(0)
            status_text = st.empty()

        # Process in batches
        for i in range(0, total_docs, BATCH_SIZE):
            batch = documents[i:i + BATCH_SIZE]
            
            if THREADING_ENABLED:
                batch_results = self._process_batch_parallel(batch)
            else:
                batch_results = self._process_batch_sequential(batch)
            
            results.extend(batch_results)
            self.processed_docs += len(batch)

            # Update progress
            if self.progress_bar:
                progress = (i + len(batch)) / total_docs
                self.progress_bar.progress(progress)
                status_text.text(f"Processed {self.processed_docs}/{total_docs} documents")

        # Clear progress indicators
        if self.progress_bar:
            self.progress_bar.empty()
            status_text.empty()

        return results

    def _process_batch_parallel(self, batch: List[Any]) -> List[Dict]:
        """Process a batch of documents in parallel."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_PROCESSING_THREADS) as executor:
            return list(executor.map(self._process_single_document, batch))

    def _process_batch_sequential(self, batch: List[Any]) -> List[Dict]:
        """Process a batch of documents sequentially."""
        return [self._process_single_document(doc) for doc in batch]

    def _process_single_document(self, doc: Any) -> Dict:
        """Process a single document with error handling."""
        try:
            # Extract metadata
            metadata = doc.metadata
            thread_id = metadata.get("thread_id")
            if thread_id:
                self.total_threads.add(thread_id)

            return {
                "thread_id": thread_id,
                "content": doc.page_content,
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return None

    def get_paginated_results(self, results: List[Dict], page: int) -> List[Dict]:
        """Get a paginated subset of results."""
        start_idx = (page - 1) * POSTS_PER_PAGE
        end_idx = start_idx + POSTS_PER_PAGE
        return results[start_idx:end_idx]

    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        return {
            "total_documents": self.processed_docs,
            "total_threads": len(self.total_threads),
            "batches_processed": ceil(self.processed_docs / BATCH_SIZE)
        }