from typing import List
from langchain_core.documents import Document

class PineconeRetriever:
    def __init__(self, index, embeddings, top_k=3):
        self.index = index
        self.embeddings = embeddings
        self.top_k = top_k
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        query_embedding = self.embeddings.embed_query(query)
        results = self.index.query(
            vector=query_embedding,
            top_k=self.top_k,
            include_metadata=True
        )
        
        return [
            Document(
                page_content=result.metadata.get("text", ""),
                metadata=result.metadata
            )
            for result in results.matches
        ]