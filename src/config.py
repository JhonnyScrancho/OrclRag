# config.py
import streamlit as st

# Constants
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # Modello corretto per 768 dimensioni
LLM_MODEL = "gpt-4-turbo-preview"
INDEX_NAME = "forum-index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200