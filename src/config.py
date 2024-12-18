# config.py
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-3.5-turbo"
INDEX_NAME = "forum-index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Pinecone Config - Updated with your specific environment
VALID_ENVIRONMENTS = [
    "us-east-1",    # Il tuo environment attuale
    "gcp-starter",
    "us-west4-gcp",
    "us-west4-gcp-free",
    "us-west1-gcp-free",
    "us-central1-gcp",
    "northamerica-northeast1-gcp",
    "asia-southeast1-gcp",
    "us-east1-gcp",
    "eu-west1-gcp",
    "eu-west4-gcp"
]