import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from config import LLM_MODEL

def setup_rag_chain(retriever):
    """Configura la catena RAG."""
    llm = ChatOpenAI(
        model_name=LLM_MODEL,
        temperature=0,
        api_key=st.secrets["OPENAI_API_KEY"]
    )
    
    template = """Usa le seguenti informazioni per rispondere alla domanda. Se non conosci la risposta, di' semplicemente che non lo sai.

    Contesto: {context}
    
    Domanda: {query}
    
    Risposta:"""
    
    prompt = PromptTemplate(template=template, input_variables=["context", "query"])
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Corretta sintassi LCEL
    chain = (
        {
            "context": lambda x: format_docs(retriever.get_relevant_documents(x)),
            "query": RunnablePassthrough()
        }
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    return chain