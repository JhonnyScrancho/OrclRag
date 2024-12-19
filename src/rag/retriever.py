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
                top_k=10000,  # Get all documents
                include_metadata=True
            )
            return results.matches
        except Exception as e:
            st.error(f"Error fetching metadata: {str(e)}")
            return []
    
    def get_database_stats(self):
        """Get comprehensive database statistics."""
        matches = self._fetch_all_metadata()
        
        # Aggregate statistics
        stats = {
            'total_posts': len(matches),
            'unique_threads': set(),
            'unique_authors': set(),
            'posts_by_thread': defaultdict(int),
            'posts_by_author': defaultdict(int),
            'keywords_frequency': defaultdict(int),
            'earliest_post': None,
            'latest_post': None,
            'threads_details': []
        }
        
        for match in matches:
            metadata = match.metadata
            thread_id = metadata.get('thread_id')
            thread_title = metadata.get('thread_title')
            author = metadata.get('author')
            post_time = metadata.get('post_time')
            keywords = metadata.get('keywords', [])
            
            # Update aggregations
            if thread_id and thread_title:
                stats['unique_threads'].add((thread_id, thread_title))
                stats['posts_by_thread'][thread_title] += 1
            
            if author:
                stats['unique_authors'].add(author)
                stats['posts_by_author'][author] += 1
            
            for keyword in keywords:
                stats['keywords_frequency'][keyword] += 1
            
            # Track thread details
            thread_info = {
                'title': thread_title,
                'url': metadata.get('url'),
                'total_posts': stats['posts_by_thread'][thread_title],
                'unique_authors': set([author]),
                'keywords': set(keywords)
            }
            
            if thread_info not in stats['threads_details']:
                stats['threads_details'].append(thread_info)
        
        return stats
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents based on query type."""
        try:
            # If query is about database statistics/content
            if any(keyword in query.lower() for keyword in ['quanti', 'numero', 'cosa parlano', 'argomenti', 'temi', 'statistic']):
                stats = self.get_database_stats()
                
                # Create comprehensive analysis document
                analysis = f"""Analisi completa del database:

Statistiche Generali:
- Totale posts: {stats['total_posts']}
- Thread unici: {len(stats['unique_threads'])}
- Autori unici: {len(stats['unique_authors'])}

Threads principali:
"""
                
                for thread in stats['threads_details']:
                    analysis += f"""
• {thread['title']}
  - Posts totali: {thread['total_posts']}
  - Autori unici: {len(thread['unique_authors'])}
  - Keywords principali: {', '.join(list(thread['keywords'])[:5])}
"""
                
                # Add keyword analysis
                top_keywords = sorted(stats['keywords_frequency'].items(), key=lambda x: x[1], reverse=True)[:10]
                analysis += "\nKeywords più frequenti:\n"
                for keyword, freq in top_keywords:
                    analysis += f"- {keyword}: {freq} occorrenze\n"
                
                return [Document(page_content=analysis, metadata={"type": "analysis"})]
            
            # For specific queries, use vector search
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