# config.py
import streamlit as st

# Dimensioni base
EMBEDDING_DIMENSION = 768  # Per Sentence Transformers mpnet
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "gpt-4-turbo-preview"
INDEX_NAME = "forum-index"

# Configurazioni per elaborazione di grandi volumi
MAX_DOCUMENTS_PER_QUERY = 1000  # Massimo numero di documenti per query
BATCH_SIZE = 100  # Dimensione del batch per il processing
MAX_TOKENS_PER_REQUEST = 120000  # Massimo numero di token per richiesta al LLM
THREADING_ENABLED = True  # Abilita il multi-threading per il processing

# Chunking settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100

# Cache settings
CACHE_TTL = 3600  # 1 hour in seconds
MAX_CACHE_ITEMS = 1000

# Rate limiting
MAX_REQUESTS_PER_MINUTE = 60
RATE_LIMIT_PERIOD = 60  # seconds

# Retrieval settings
INITIAL_RETRIEVAL_K = 1000  # Aumentato per grandi volumi
FINAL_K = 1000  # Aumentato per grandi volumi

# Performance settings
ENABLE_BATCH_PROCESSING = True
PARALLEL_PROCESSING_THREADS = 4
MEMORY_LIMIT = 8589934592  # 8GB in bytes

# Pagination settings
POSTS_PER_PAGE = 50
MAX_PAGES = 20