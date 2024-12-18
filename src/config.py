import streamlit as st

# Constants
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-3.5-turbo"
INDEX_NAME = "rag-p5eyqni"  # Nome dell'indice esistente
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Pinecone Config
PINECONE_CLOUD = "gcp"  # Cambiato da aws a gcp
PINECONE_REGION = "gcp-starter"  # Cambiato da us-east-1 a gcp-starter