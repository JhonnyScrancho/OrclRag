from typing import List
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

class PineconeRetriever:
    def __init__(self, index, embeddings, top_k=3):
        """
        Inizializza il retriever di Pinecone.
        
        Args:
            index: Indice Pinecone
            embeddings: Modello per generare embeddings
            top_k: Numero di documenti da recuperare (default: 3)
        """
        self.index = index
        self.embeddings = embeddings
        self.top_k = top_k
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Recupera i documenti più rilevanti per la query.
        
        Args:
            query: La query testuale
            
        Returns:
            List[Document]: Lista di documenti rilevanti
            
        Raises:
            ValueError: Se ci sono problemi con la query o gli embeddings
            Exception: Per altri errori durante il recupero
        """
        try:
            if not query or not isinstance(query, str):
                raise ValueError("Query non valida")
            
            # Genera l'embedding della query
            query_embedding = self.embeddings.embed_query(query)
            
            # Interroga Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=self.top_k,
                include_metadata=True
            )
            
            # Converte i risultati in documenti Langchain
            documents = []
            for result in results.matches:
                if not result.metadata.get("text"):
                    logger.warning(f"Documento senza testo trovato: {result.id}")
                    continue
                    
                doc = Document(
                    page_content=result.metadata.get("text", ""),
                    metadata={
                        "thread_title": result.metadata.get("thread_title", ""),
                        "url": result.metadata.get("url", ""),
                        "timestamp": result.metadata.get("timestamp", ""),
                        "score": result.score
                    }
                )
                documents.append(doc)
            
            if not documents:
                logger.warning("Nessun documento rilevante trovato per la query")
                
            return documents
            
        except ValueError as e:
            logger.error(f"Errore di validazione: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Errore nel recupero dei documenti: {str(e)}", exc_info=True)
            raise ValueError(f"Impossibile recuperare i documenti: {str(e)}")