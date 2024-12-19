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
        unique_threads = {}
        post_quotes = []
        
        for match in matches:
            metadata = match.metadata
            thread_id = metadata.get('thread_id')
            thread_title = metadata.get('thread_title')
            
            # Track unique threads
            if thread_id and thread_title and thread_id not in unique_threads:
                unique_threads[thread_id] = {
                    'title': thread_title,
                    'url': metadata.get('url'),
                    'total_posts': metadata.get('total_posts', 0),
                    'scrape_time': metadata.get('scrape_time')
                }
            
            # Track quotes
            if metadata.get('quoted_author'):
                post_quotes.append({
                    'quoted_author': metadata.get('quoted_author'),
                    'quoted_content': metadata.get('quoted_content'),
                    'post_author': metadata.get('author')
                })
        
        return {
            'threads': unique_threads,
            'total_threads': len(unique_threads),
            'total_posts': len(matches),
            'quotes': post_quotes
        }
    
    def get_thread_summary(self, thread_id=None):
        """Generate a summary of thread content."""
        matches = self._fetch_all_metadata()
        
        # Get the first thread if none specified
        if not thread_id and matches:
            thread_id = matches[0].metadata.get('thread_id')
        
        thread_posts = []
        thread_info = {}
        
        for match in matches:
            metadata = match.metadata
            if metadata.get('thread_id') == thread_id:
                if not thread_info:
                    thread_info = {
                        'title': metadata.get('thread_title'),
                        'url': metadata.get('url'),
                        'total_posts': metadata.get('total_posts')
                    }
                
                post_content = {
                    'author': metadata.get('author'),
                    'content': metadata.get('text', '').split('Content: ')[-1].split('Keywords:')[0].strip(),
                    'post_time': metadata.get('post_time'),
                    'quoted_author': metadata.get('quoted_author'),
                    'quoted_content': metadata.get('quoted_content')
                }
                thread_posts.append(post_content)
        
        return thread_info, thread_posts
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents based on query type."""
        try:
            # Handle specific query types
            if any(keyword in query.lower() for keyword in ['quanti', 'numero', 'thread', 'post']):
                stats = self.get_database_stats()
                
                analysis = f"""Statistiche del Database:
- Numero di thread: {stats['total_threads']}
- Numero totale di post: {stats['total_posts']}
"""
                return [Document(page_content=analysis, metadata={"type": "analysis"})]
            
            # Handle quote queries
            elif 'cit' in query.lower() or 'quot' in query.lower():
                stats = self.get_database_stats()
                quotes_text = "Citazioni nei post:\n\n"
                for quote in stats['quotes']:
                    quotes_text += f"- {quote['post_author']} cita {quote['quoted_author']}: {quote['quoted_content']}\n\n"
                return [Document(page_content=quotes_text, metadata={"type": "quotes"})]
            
            # Handle thread summary queries
            elif 'riassunto' in query.lower() or 'parlano' in query.lower():
                thread_info, posts = self.get_thread_summary()
                if thread_info:
                    summary = f"""Thread: {thread_info['title']}
URL: {thread_info['url']}
Totale posts: {thread_info['total_posts']}

Contenuto principale:
"""
                    # Add first post content as it usually contains the main topic
                    if posts:
                        summary += f"\n{posts[0]['content']}\n"
                        
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