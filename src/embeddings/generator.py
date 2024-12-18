import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from config import CHUNK_SIZE, CHUNK_OVERLAP

def create_chunks(texts: list[str]) -> list:
    """Divide i testi in chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.create_documents(texts)

def get_embeddings():
    """Inizializza il modello di embeddings."""
    from openai import OpenAI
    
    client = OpenAI(
        api_key=st.secrets["OPENAI_API_KEY"]
    )
    
    embeddings = OpenAIEmbeddings(
        api_key=st.secrets["OPENAI_API_KEY"]
    )
    
    return embeddings