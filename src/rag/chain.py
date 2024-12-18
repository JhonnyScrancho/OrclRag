from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from config import LLM_MODEL

def setup_rag_chain(retriever):
    """Configura la catena RAG."""
    llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0)
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return chain