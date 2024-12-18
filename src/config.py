import streamlit as st

# Constants
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-3.5-turbo"
INDEX_NAME = "rag"  # Nome dell'indice esistente
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Pinecone Serverless Config
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"