# generator.py
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_DIMENSION
import torch
import numpy as np

class SentenceTransformersEmbeddings:
    def __init__(self, model_name="paraphrase-multilingual-mpnet-base-v2"):
        st.write(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        if torch.cuda.is_available():
            self.model.to('cuda')
        self.dimension = self.model.get_sentence_embedding_dimension()
        st.write(f"Model loaded. Embedding dimension: {self.dimension}")

    def embed_query(self, text):
        """Genera embedding per una singola query."""
        with torch.no_grad():
            embedding = self.model.encode(text, normalize_embeddings=True)
            # Convert to list and ensure correct dimension
            embedding = embedding.tolist()
            st.write(f"Generated embedding dimension: {len(embedding)}")
            if len(embedding) != 768:
                raise ValueError(f"Expected embedding dimension 768, got {len(embedding)}")
            return embedding

def create_chunks(texts: list[str]) -> list:
    """Divide i testi in chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.create_documents(texts)

def get_embeddings():
    """Inizializza il modello di embeddings."""
    return SentenceTransformersEmbeddings()