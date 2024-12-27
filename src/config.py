# config.py
import streamlit as st

# Constants
EMBEDDING_DIMENSION = 768  # Dimensione per Sentence Transformers mpnet
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
LLM_MODEL = "gpt-4-turbo-preview"
INDEX_NAME = "forum-index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200