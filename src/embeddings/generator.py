from sentence_transformers import SentenceTransformer
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Any
import logging
from datetime import datetime
from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_DIMENSION, EMBEDDING_MODEL

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
                logger.info(f"Generating embedding for text of length: {len(text)}")
                
                embedding = self.model.encode(text, normalize_embeddings=True)
                
                if len(embedding) != self.dimension:
                    raise ValueError(f"Dimensione embedding non corretta. Attesa {self.dimension}, ricevuta {len(embedding)}")
                
                logger.info(f"Successfully generated embedding of dimension: {len(embedding)}")
                return embedding.tolist()
                
        except Exception as e:
            logger.error(f"Errore generazione embedding: {str(e)}")
            raise

def extract_metadata(text: str) -> Dict[str, Any]:
    """Estrae i metadati dal testo del post."""
    metadata = {}
    
    # Estrai le informazioni base
    lines = text.split('\n')
    for line in lines:
        if line.startswith('Author: '):
            metadata['author'] = line.replace('Author: ', '').strip()
        elif line.startswith('Time: '):
            metadata['post_time'] = line.replace('Time: ', '').strip()
            try:
                # Standardizza il formato della data
                dt = datetime.fromisoformat(metadata['post_time'])
                metadata['post_time'] = dt.isoformat()
            except ValueError:
                pass
        elif line.startswith('Keywords: '):
            metadata['keywords'] = [k.strip() for k in line.replace('Keywords: ', '').split(',')]
        elif line.startswith('Sentiment: '):
            try:
                metadata['sentiment'] = float(line.replace('Sentiment: ', '').strip())
            except ValueError:
                metadata['sentiment'] = 0.0
                
    return metadata

def create_chunks(texts: List[str]) -> List[Document]:
    """Divide i testi in chunks mantenendo i metadati."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True
    )
    
    chunks = []
    for text in texts:
        # Estrai i metadati prima del chunking
        metadata = extract_metadata(text)
        
        # Crea i chunks mantenendo i metadati
        doc_chunks = text_splitter.create_documents([text])
        for i, chunk in enumerate(doc_chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_number": i,
                "total_chunks": len(doc_chunks),
                "text": text,  # Mantieni il testo originale completo
                "chunk_text": chunk.page_content  # Il testo del chunk specifico
            })
            chunks.append(Document(
                page_content=chunk.page_content,
                metadata=chunk_metadata
            ))
            
    logger.info(f"Created {len(chunks)} chunks from {len(texts)} texts")
    return chunks

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