import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from config import CHUNK_SIZE, CHUNK_OVERLAP
from typing import List
from openai import OpenAI
import numpy as np

class ModernOpenAIEmbeddings(Embeddings):
    """Classe per gli embeddings che usa il client OpenAI più recente."""
    
    def __init__(self):
        # Inizializza il client una sola volta
        self.client = OpenAI(
            api_key=st.secrets["OPENAI_API_KEY"]
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Genera embeddings per una lista di testi."""
        try:
            # Gestisce le richieste in batch
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            st.error(f"Errore durante la generazione degli embeddings: {str(e)}")
            raise e
    
    def embed_query(self, text: str) -> List[float]:
        """Genera embedding per un singolo testo di query."""
        try:
            if not isinstance(text, str):
                text = str(text)
                
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            st.error(f"Errore durante la generazione dell'embedding per la query: {str(e)}")
            raise e

def create_chunks(texts: list[str]) -> list:
    """Divide i testi in chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.create_documents(texts)

def get_embeddings():
    """Inizializza il modello di embeddings."""
    try:
        return ModernOpenAIEmbeddings()
    except Exception as e:
        st.error(f"Errore durante l'inizializzazione degli embeddings: {str(e)}")
        raise e