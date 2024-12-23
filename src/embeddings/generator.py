# generator.py
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_DIMENSION
import torch
import logging

# Configura logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class SentenceTransformersEmbeddings:
    def __init__(self, model_name="paraphrase-multilingual-mpnet-base-v2"):
        logger.debug(f"Initializing SentenceTransformer with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Log delle dimensioni del modello
        actual_dim = self.model.get_sentence_embedding_dimension()
        logger.debug(f"Model's actual dimension: {actual_dim}")
        logger.debug(f"Expected dimension from config: {EMBEDDING_DIMENSION}")
        
        if actual_dim != EMBEDDING_DIMENSION:
            logger.warning(f"Dimension mismatch! Model: {actual_dim}, Config: {EMBEDDING_DIMENSION}")
        
        # Se disponibile, usa la GPU
        if torch.cuda.is_available():
            logger.debug("CUDA available - moving model to GPU")
            self.model.to('cuda')
        
        self.dimension = actual_dim  # Usa la dimensione effettiva del modello

    def embed_query(self, text):
        """Genera embedding per una singola query."""
        logger.debug("Generating embedding for query")
        with torch.no_grad():
            embedding = self.model.encode(text, normalize_embeddings=True)
            logger.debug(f"Generated embedding dimension: {len(embedding)}")
            return embedding.tolist()

    def embed_documents(self, documents):
        """Genera embeddings per una lista di documenti."""
        logger.debug(f"Generating embeddings for {len(documents)} documents")
        with torch.no_grad():
            embeddings = self.model.encode(documents, normalize_embeddings=True)
            logger.debug(f"Generated embeddings shape: {embeddings.shape}")
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
    logger.debug("Initializing embeddings model")
    embeddings = SentenceTransformersEmbeddings()
    
    # Test embedding
    test_embedding = embeddings.embed_query("test sentence")
    logger.debug(f"Test embedding dimension: {len(test_embedding)}")
    
    return embeddings