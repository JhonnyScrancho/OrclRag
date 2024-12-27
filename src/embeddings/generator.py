# generator.py

from sentence_transformers import SentenceTransformer
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_DIMENSION, EMBEDDING_MODEL
import logging

logger = logging.getLogger(__name__)

class SentenceTransformersEmbeddings:
    def __init__(self, model_name=EMBEDDING_MODEL):
        try:
            self.model = SentenceTransformer(model_name)
            # Valida la dimensione del modello
            test_embedding = self.model.encode("test", normalize_embeddings=True)
            actual_dimension = len(test_embedding)
            
            logger.info(f"Model loaded with dimension: {actual_dimension}")
            
            if actual_dimension != EMBEDDING_DIMENSION:
                raise ValueError(f"Dimensione del modello non corretta. Attesa {EMBEDDING_DIMENSION}, ricevuta {actual_dimension}")
                
            self.dimension = EMBEDDING_DIMENSION
            
            # Usa GPU se disponibile
            if torch.cuda.is_available():
                self.model.to('cuda')
                logger.info("Using GPU for embeddings")
            else:
                logger.info("Using CPU for embeddings")
                
        except Exception as e:
            logger.error(f"Errore inizializzazione embeddings: {str(e)}")
            raise

    def embed_query(self, text):
        """Genera embedding per una singola query."""
        try:
            with torch.no_grad():
                # Aggiungi log per debugging
                logger.info(f"Generating embedding for text of length: {len(text)}")
                
                embedding = self.model.encode(text, normalize_embeddings=True)
                
                # Verifica dimensione
                if len(embedding) != self.dimension:
                    raise ValueError(f"Dimensione embedding non corretta. Attesa {self.dimension}, ricevuta {len(embedding)}")
                
                logger.info(f"Successfully generated embedding of dimension: {len(embedding)}")
                return embedding.tolist()
                
        except Exception as e:
            logger.error(f"Errore generazione embedding: {str(e)}")
            raise

def create_chunks(texts):
    """Divide i testi in chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]  # Migliore gestione dei separatori
    )
    chunks = text_splitter.create_documents(texts)
    
    # Aggiungi logging per debug
    logger.info(f"Created {len(chunks)} chunks from {len(texts)} texts")
    for i, chunk in enumerate(chunks):
        logger.debug(f"Chunk {i}: {len(chunk.page_content)} chars")
    
    return chunks

def remove_duplicates(index):
    """Rimuove eventuali duplicati dal database."""
    try:
        # Query per tutti i documenti
        results = index.query(
            vector=[0.0] * EMBEDDING_DIMENSION,
            top_k=10000,
            include_metadata=True
        )
        
        seen = {}
        duplicates = []
        for match in results.matches:
            key = f"{match.metadata['thread_id']}_{match.metadata['post_id']}"
            if key in seen:
                duplicates.append(match.id)
            seen[key] = True
            
        if duplicates:
            index.delete(ids=duplicates)
            logger.info(f"Removed {len(duplicates)} duplicate vectors")
    except Exception as e:
        logger.error(f"Error removing duplicates: {str(e)}")

def get_embeddings():
    """Inizializza il modello embeddings con logging dettagliato."""
    try:
        logger.info("Initializing embeddings model...")
        embeddings = SentenceTransformersEmbeddings()
        logger.info(f"Successfully initialized embeddings with dimension: {embeddings.dimension}")
        return embeddings
    except Exception as e:
        logger.error(f"Fatal error initializing embeddings: {str(e)}")
        raise