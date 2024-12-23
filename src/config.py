# config.py
import streamlit as st

# Constants
EMBEDDING_DIMENSION = 768  # Dimensione per Sentence Transformers mpnet
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "gpt-4-turbo-preview"
INDEX_NAME = "forum-index"

# Chunking settings
CHUNK_SIZE = 512  # Ridotto per migliore granularità
CHUNK_OVERLAP = 100  # Ridotto proporzionalmente

# Cache settings
CACHE_TTL = 3600  # 1 hour in seconds
MAX_CACHE_ITEMS = 1000

# Rate limiting
MAX_REQUESTS_PER_MINUTE = 60
RATE_LIMIT_PERIOD = 60  # seconds

# Re-ranking settings
INITIAL_RETRIEVAL_K = 20  # Number of documents to retrieve before re-ranking
FINAL_K = 10  # Number of documents after re-ranking