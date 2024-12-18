import streamlit as st

# Constants
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-3.5-turbo"
INDEX_NAME = "forum-index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Pinecone Config
VALID_ENVIRONMENTS = [
    "us-east1-aws",
    "us-west1-aws",
    "asia-southeast1-aws",
    "eu-west1-aws"
]