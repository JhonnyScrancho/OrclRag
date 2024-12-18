import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from config import CHUNK_SIZE, CHUNK_OVERLAP
from typing import List
import openai

class DirectOpenAIEmbeddings(Embeddings):
    """Classe minimalista per gli embeddings che usa l'API OpenAI di base."""
    
    def __init__(self):
        openai.api_key = st.secrets["OPENAI_API_KEY"]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Genera embeddings per una lista di testi."""
        embeddings = []
        for text in texts:
            response = openai.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            embeddings.append(response.data[0].embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Genera embedding per un singolo testo di query."""
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding

def create_chunks(texts: list[str]) -> list:
    """Divide i testi in chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.create_documents(texts)

def get_embeddings():
    """Inizializza il modello di embeddings."""
    return DirectOpenAIEmbeddings()