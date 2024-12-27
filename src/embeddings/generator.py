from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict, Optional, Union
from config import EMBEDDING_DIMENSION, EMBEDDING_MODEL
import logging
import numpy as np

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
                self.model = self.model.to('cuda')
                logger.info("Using GPU for embeddings")
            else:
                logger.info("Using CPU for embeddings")
                
        except Exception as e:
            logger.error(f"Errore inizializzazione embeddings: {str(e)}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Genera embedding per una singola query."""
        try:
            if not text or not isinstance(text, str):
                logger.warning("Invalid text input for embedding")
                return [0.0] * self.dimension
                
            with torch.no_grad():
                # Aggiungi log per debugging
                logger.debug(f"Generating embedding for text of length: {len(text)}")
                
                # Limita lunghezza massima se necessario
                max_length = 512
                if len(text) > max_length:
                    logger.warning(f"Text truncated from {len(text)} to {max_length} characters")
                    text = text[:max_length]
                
                embedding = self.model.encode(text, normalize_embeddings=True)
                
                # Verifica dimensione
                if len(embedding) != self.dimension:
                    raise ValueError(f"Dimensione embedding non corretta. Attesa {self.dimension}, ricevuta {len(embedding)}")
                
                logger.debug(f"Successfully generated embedding of dimension: {len(embedding)}")
                return embedding.tolist()
                
        except Exception as e:
            logger.error(f"Errore generazione embedding: {str(e)}")
            raise

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Genera embeddings per un batch di testi."""
        try:
            if not texts:
                logger.warning("Empty text list provided for batch embedding")
                return []
                
            with torch.no_grad():
                all_embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    
                    # Filtra testi non validi
                    valid_texts = [text for text in batch if text and isinstance(text, str)]
                    if not valid_texts:
                        continue
                        
                    # Genera embeddings
                    batch_embeddings = self.model.encode(valid_texts, normalize_embeddings=True)
                    
                    # Gestisci output singolo vs multiplo
                    if len(valid_texts) == 1:
                        batch_embeddings = [batch_embeddings]
                    
                    all_embeddings.extend([emb.tolist() for emb in batch_embeddings])
                    
                    logger.debug(f"Processed batch {i//batch_size + 1} with {len(valid_texts)} texts")
                
                return all_embeddings
                
        except Exception as e:
            logger.error(f"Errore generazione batch embeddings: {str(e)}")
            raise

    def combine_embeddings(self, embeddings: List[List[float]], weights: Optional[List[float]] = None) -> List[float]:
        """Combina più embeddings in uno singolo."""
        try:
            if not embeddings:
                return [0.0] * self.dimension
                
            if weights is None:
                weights = [1.0] * len(embeddings)
                
            if len(embeddings) != len(weights):
                raise ValueError("Number of embeddings must match number of weights")
                
            # Combina con pesi
            weighted_sum = np.zeros(self.dimension)
            for emb, weight in zip(embeddings, weights):
                weighted_sum += np.array(emb) * weight
                
            # Normalizza
            norm = np.linalg.norm(weighted_sum)
            if norm > 0:
                weighted_sum = weighted_sum / norm
                
            return weighted_sum.tolist()
            
        except Exception as e:
            logger.error(f"Error combining embeddings: {str(e)}")
            raise

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

def validate_embedding(embedding: List[float], dimension: int = EMBEDDING_DIMENSION) -> bool:
    """Valida un embedding."""
    try:
        if not isinstance(embedding, list):
            return False
        if len(embedding) != dimension:
            return False
            
        # Verifica che tutti gli elementi siano numeri
        if not all(isinstance(x, (int, float)) for x in embedding):
            return False
            
        # Verifica che non ci siano NaN o Inf
        if any(np.isnan(x) or np.isinf(x) for x in embedding):
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error validating embedding: {str(e)}")
        return False

def normalize_embedding(embedding: List[float]) -> List[float]:
    """Normalizza un embedding."""
    try:
        arr = np.array(embedding)
        norm = np.linalg.norm(arr)
        if norm > 0:
            return (arr / norm).tolist()
        return embedding
    except Exception as e:
        logger.error(f"Error normalizing embedding: {str(e)}")
        return embedding

def process_text_for_embedding(text: str, max_length: int = 512) -> str:
    """Prepara il testo per l'embedding."""
    try:
        if not text:
            return ""
            
        # Rimuovi caratteri non necessari
        text = text.replace("\n", " ").replace("\r", " ")
        
        # Rimuovi spazi multipli
        text = " ".join(text.split())
        
        # Tronca se necessario
        if len(text) > max_length:
            text = text[:max_length]
            
        return text
        
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        return ""

class BatchProcessor:
    """Classe per il processing batch di testi ed embedding."""
    
    def __init__(self, embeddings_model: SentenceTransformersEmbeddings, batch_size: int = 32):
        self.embeddings = embeddings_model
        self.batch_size = batch_size
        
    def process_batch(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """Processa un batch di testi."""
        try:
            if not texts:
                return []
                
            # Prepara i testi
            processed_texts = [
                process_text_for_embedding(text)
                for text in texts
                if text
            ]
            
            if not processed_texts:
                return []
                
            # Genera embeddings
            embeddings = self.embeddings.embed_batch(
                processed_texts,
                batch_size=self.batch_size
            )
            
            # Valida risultati
            valid_embeddings = [
                emb for emb in embeddings
                if validate_embedding(emb)
            ]
            
            if len(valid_embeddings) != len(texts):
                logger.warning(f"Some embeddings were invalid: {len(valid_embeddings)}/{len(texts)}")
                
            return valid_embeddings
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            return []
            
    def get_document_embeddings(self, documents: List[Dict], text_key: str = "text") -> List[Dict]:
        """Genera embeddings per una lista di documenti."""
        try:
            texts = [doc.get(text_key, "") for doc in documents]
            embeddings = self.process_batch(texts)
            
            # Combina documenti con i loro embeddings
            result = []
            for doc, emb in zip(documents, embeddings):
                if validate_embedding(emb):
                    doc_copy = doc.copy()
                    doc_copy["embedding"] = emb
                    result.append(doc_copy)
                    
            return result
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            return []