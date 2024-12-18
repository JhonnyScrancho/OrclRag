import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from openai import OpenAI
from config import CHUNK_SIZE, CHUNK_OVERLAP
from typing import List

class CustomOpenAIEmbeddings(Embeddings):
    """Classe personalizzata per gli embeddings che usa direttamente il client OpenAI."""
    
    def __init__(self):
        self.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Genera embeddings per una lista di testi."""
        embeddings = []
        # Process in batches to avoid rate limits
        for text in texts:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            embeddings.append(response.data[0].embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Genera embedding per un singolo testo di query."""
        response = self.client.embeddings.create(
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
    return CustomOpenAIEmbeddings()