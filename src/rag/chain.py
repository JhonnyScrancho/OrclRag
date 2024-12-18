import openai
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from config import LLM_MODEL

def setup_rag_chain(retriever):
    """Configura la catena RAG."""
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    
    llm = ChatOpenAI(
        model_name=LLM_MODEL,
        temperature=0,
        openai_api_key=st.secrets["OPENAI_API_KEY"]
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return chain