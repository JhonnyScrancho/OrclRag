from typing import List
import streamlit as st
from config import INDEX_NAME

def ensure_index_exists(pinecone):
    """Assicura che l'indice Pinecone esista."""
    # Get list of existing indexes
    existing_indexes = pinecone.list_indexes()
    
    # Create index if it doesn't exist
    if INDEX_NAME not in existing_indexes:
        pinecone.create_index(
            name=INDEX_NAME,
            dimension=1536,  # dimensione per OpenAI embeddings
            metric="cosine"
        )
    
    # Get the index
    return pinecone.Index(INDEX_NAME)

def update_document_in_index(index, doc_id: str, embedding: List[float], metadata: dict):
    """Aggiorna o inserisce un documento nell'indice."""
    index.upsert(vectors=[(doc_id, embedding, metadata)])