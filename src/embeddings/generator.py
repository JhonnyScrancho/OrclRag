# generator.py
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_DIMENSION
import numpy as np
import torch

class SentenceTransformersEmbeddings:
    def __init__(self, model_name="paraphrase-multilingual-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
        # Se disponibile, usa la GPU
        if torch.cuda.is_available():
            self.model.to('cuda')
        self.dimension = EMBEDDING_DIMENSION

    def embed_query(self, text):
        """Genera embedding per una singola query."""
        with torch.no_grad():
            embedding = self.model.encode(text, normalize_embeddings=True)
            if len(embedding) != EMBEDDING_DIMENSION:  # Verifica dimensione
                raise ValueError(f"L'embedding ha dimensione {len(embedding)}, attesa {EMBEDDING_DIMENSION}")
            return embedding.tolist()

    def embed_documents(self, documents):
        """Genera embeddings per una lista di documenti."""
        with torch.no_grad():
            embeddings = self.model.encode(documents, normalize_embeddings=True)
            for emb in embeddings:
                if len(emb) != EMBEDDING_DIMENSION:  # Verifica dimensione
                    raise ValueError(f"L'embedding ha dimensione {len(emb)}, attesa {EMBEDDING_DIMENSION}")
            return [emb.tolist() for emb in embeddings]

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