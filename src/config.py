# Modelli
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-3.5-turbo"

# Index settings
INDEX_NAME = "forum-index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Pinecone Config
VALID_ENVIRONMENTS = [
    "us-east-1",  # Environment corrente dall'interfaccia Pinecone
]

def validate_pinecone_environment(environment: str) -> bool:
    """Valida il formato dell'environment di Pinecone."""
    return environment in VALID_ENVIRONMENTS