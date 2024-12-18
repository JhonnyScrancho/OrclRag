import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from config import CHUNK_SIZE, CHUNK_OVERLAP
import os

def create_chunks(texts: list[str]) -> list:
    """Divide i testi in chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.create_documents(texts)

def get_embeddings():
    """Inizializza il modello di embeddings."""
    # Ensure no proxy settings are inherited
    if 'http_proxy' in os.environ:
        del os.environ['http_proxy']
    if 'https_proxy' in os.environ:
        del os.environ['https_proxy']
    
    return OpenAIEmbeddings(
        openai_api_key=st.secrets["OPENAI_API_KEY"],
        model="text-embedding-ada-002",
        client=None  # Forces creation of a new client without proxy settings
    )