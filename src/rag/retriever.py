import streamlit as st
from typing import List, Dict
from langchain_core.documents import Document
from collections import defaultdict

class PineconeRetriever:
    def __init__(self, index, embeddings):
        self.index = index
        self.embeddings = embeddings
        self._metadata_cache = None
    
    def _fetch_all_metadata(self):
        """Fetch and cache all metadata from the index."""
        try:
            results = self.index.query(
                vector=[0] * 1536,
                top_k=10000,
                include_metadata=True
            )
            return results.matches
        except Exception as e:
            st.error(f"Error fetching metadata: {str(e)}")
            return []

    def get_database_stats(self):
        """Get comprehensive database statistics."""
        matches = self._fetch_all_metadata()
        
        # Initialize thread tracking
        thread_data = {}
        quotes_count = 0
        
        # Group by thread
        for match in matches:
            metadata = match.metadata
            thread_id = metadata.get('thread_id')
            
            if thread_id not in thread_data:
                thread_data[thread_id] = {
                    'title': metadata.get('thread_title'),
                    'url': metadata.get('url'),
                    'posts': [],
                    'total_posts': metadata.get('total_posts', 0)
                }
            
            # Count quotes by looking for 'said:' pattern in content
            content = metadata.get('text', '')
            if ' said:' in content and 'Click to expand...' in content:
                quotes_count += 1
            
            thread_data[thread_id]['posts'].append(metadata)
        
        return {
            'total_threads': len(thread_data),
            'total_posts': len(matches),
            'quotes_count': quotes_count,
            'threads': thread_data
        }

    def get_thread_summary(self):
        """Generate a comprehensive thread summary."""
        matches = self._fetch_all_metadata()
        if not matches:
            return None
        
        # Get thread info from first match
        first_match = matches[0]
        thread_info = {
            'title': first_match.metadata.get('thread_title'),
            'url': first_match.metadata.get('url'),
            'total_posts': first_match.metadata.get('total_posts'),
            'first_post_content': ''
        }
        
        # Find the first post (original post) of the thread
        for match in matches:
            content = match.metadata.get('text', '')
            # Skip if it's a quote/reply
            if ' said:' not in content:
                # Extract just the content part
                content_parts = content.split('Content: ')
                if len(content_parts) > 1:
                    thread_info['first_post_content'] = content_parts[1].split('Keywords:')[0].strip()
                break
        
        return thread_info

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents based on query type."""
        try:
            # Handle statistics queries
            if any(keyword in query.lower() for keyword in ['quanti', 'numero', 'thread', 'post']):
                stats = self.get_database_stats()
                analysis = f"Il database contiene {stats['total_threads']} thread e un totale di {stats['total_posts']} post."
                return [Document(page_content=analysis, metadata={"type": "analysis"})]
            
            # Handle quote queries
            elif any(keyword in query.lower() for keyword in ['citazioni', 'quote', 'citano']):
                stats = self.get_database_stats()
                quotes_text = f"Nel database sono presenti un totale di {stats['quotes_count']} citazioni / quote."
                return [Document(page_content=quotes_text, metadata={"type": "quotes"})]
            
            # Handle summary queries
            elif any(keyword in query.lower() for keyword in ['riassunto', 'parlano', 'discussione']):
                thread_info = self.get_thread_summary()
                if thread_info:
                    summary = f"""Thread: {thread_info['title']}
URL: {thread_info['url']}

Riassunto:
Il thread discute {thread_info['first_post_content']}

Il thread contiene un totale di {thread_info['total_posts']} post con discussioni e risposte tra gli utenti."""
                    
                    return [Document(page_content=summary, metadata={"type": "summary"})]
            
            # For other queries, use vector search
            query_embedding = self.embeddings.embed_query(query)
            results = self.index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True
            )
            
            return [
                Document(
                    page_content=result.metadata.get("text", ""),
                    metadata=result.metadata
                )
                for result in results.matches
            ]
            
        except Exception as e:
            st.error(f"Error in retrieval: {str(e)}")
            return []