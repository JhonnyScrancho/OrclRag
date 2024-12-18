import pinecone
from typing import List
import streamlit as st
from src.config import INDEX_NAME

def ensure_index_exists():
    """Assicura che l'indice Pinecone esista."""
    if INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(
            name=INDEX_NAME,
            dimension=1536,  # dimensione per OpenAI embeddings
            metric="cosine"
        )
    return pinecone.Index(INDEX_NAME)

def update_document_in_index(index, doc_id: str, embedding: List[float], metadata: dict):
    """Aggiorna o inserisce un documento nell'indice."""
    index.upsert(vectors=[(doc_id, embedding, metadata)])