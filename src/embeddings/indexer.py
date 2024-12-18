from typing import List
import streamlit as st
from config import INDEX_NAME
from pinecone import Pinecone, ServerlessSpec

def ensure_index_exists(pc):
    """Assicura che l'indice Pinecone esista."""
    # Get list of existing indexes
    indexes = pc.list_indexes()
    
    # Create index if it doesn't exist
    if INDEX_NAME not in [index.name for index in indexes]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,  # dimensione per OpenAI embeddings
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-west-2"
            )
        )
    
    # Get the index
    return pc.Index(INDEX_NAME)

def update_document_in_index(index, doc_id: str, embedding: List[float], metadata: dict):
    """Aggiorna o inserisce un documento nell'indice."""
    index.upsert(vectors=[(doc_id, embedding, metadata)])