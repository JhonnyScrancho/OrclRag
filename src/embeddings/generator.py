from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_DIMENSION, EMBEDDING_MODEL
import logging

logger = logging.getLogger(__name__)

class SentenceTransformersEmbeddings:
    def __init__(self, model_name=EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        # Validate model output dimension matches configuration
        test_embedding = self.model.encode("test", normalize_embeddings=True)
        actual_dimension = len(test_embedding)
        if actual_dimension != EMBEDDING_DIMENSION:
            logger.error(f"Model dimension mismatch. Expected {EMBEDDING_DIMENSION}, got {actual_dimension}")
            raise ValueError(f"Model dimension mismatch. Expected {EMBEDDING_DIMENSION}, got {actual_dimension}")
            
        self.dimension = EMBEDDING_DIMENSION
        # Use GPU if available
        if torch.cuda.is_available():
            self.model.to('cuda')
        logger.info(f"Initialized embeddings model with dimension {self.dimension}")

    def embed_query(self, text):
        """Generate embedding for a single query."""
        try:
            with torch.no_grad():
                embedding = self.model.encode(text, normalize_embeddings=True)
                # Validate dimension
                if len(embedding) != self.dimension:
                    raise ValueError(f"Embedding dimension mismatch. Expected {self.dimension}, got {len(embedding)}")
                return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise

    def embed_documents(self, documents):
        """Generate embeddings for a list of documents."""
        try:
            with torch.no_grad():
                embeddings = self.model.encode(documents, normalize_embeddings=True)
                # Validate dimensions
                if embeddings.shape[1] != self.dimension:
                    raise ValueError(f"Embeddings dimension mismatch. Expected {self.dimension}, got {embeddings.shape[1]}")
                return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Error generating document embeddings: {str(e)}")
            raise

def create_chunks(texts: list[str]) -> list:
    """Split texts into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.create_documents(texts)

def get_embeddings():
    """Initialize embeddings model with proper configuration."""
    try:
        return SentenceTransformersEmbeddings()
    except Exception as e:
        logger.error(f"Error initializing embeddings: {str(e)}")
        raise