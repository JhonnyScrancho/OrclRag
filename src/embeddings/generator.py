import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
import logging

logger = logging.getLogger(__name__)

def create_chunks(texts: list[str]) -> list:
    """Divide i testi in chunks."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        return text_splitter.create_documents(texts)
    except Exception as e:
        logger.error(f"Errore nella creazione dei chunks: {str(e)}", exc_info=True)
        raise

def get_embeddings():
    """Inizializza il modello di embeddings."""
    try:
        from openai import OpenAI
        
        if "OPENAI_API_KEY" not in st.secrets:
            raise ValueError("API key di OpenAI non trovata nei secrets")
        
        client = OpenAI(
            api_key=st.secrets["OPENAI_API_KEY"]
        )
        
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=st.secrets["OPENAI_API_KEY"]
        )
        
        return embeddings
    except ImportError:
        logger.error("Errore: openai package non installato")
        raise
    except Exception as e:
        logger.error(f"Errore nell'inizializzazione del modello di embeddings: {str(e)}", exc_info=True)
        raise