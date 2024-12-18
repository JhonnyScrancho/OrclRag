import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from config import LLM_MODEL
import logging

logger = logging.getLogger(__name__)

def setup_rag_chain(retriever):
    """Configura la catena RAG."""
    try:
        if "OPENAI_API_KEY" not in st.secrets:
            raise ValueError("API key di OpenAI non trovata nei secrets")
        
        llm = ChatOpenAI(
            model_name=LLM_MODEL,
            temperature=0,
            api_key=st.secrets["OPENAI_API_KEY"]
        )
        
        template = """Sei un assistente esperto che analizza i contenuti di un forum.
        Usa le seguenti informazioni per rispondere alla domanda in modo dettagliato e accurato.
        Se non sei sicuro o non hai abbastanza informazioni, dillo chiaramente.

        Contesto: {context}
        
        Domanda: {query}
        
        Risposta:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "query"])
        
        def format_docs(docs):
            try:
                return "\n\n".join(doc.page_content for doc in docs)
            except Exception as e:
                logger.error(f"Errore nella formattazione dei documenti: {str(e)}")
                raise
        
        chain = (
            {"context": retriever.get_relevant_documents | format_docs, 
             "query": RunnablePassthrough()}
            | prompt 
            | llm 
            | StrOutputParser()
        )
        
        return chain
        
    except Exception as e:
        logger.error(f"Errore nella configurazione della catena RAG: {str(e)}", exc_info=True)
        raise ValueError(f"Impossibile configurare la catena RAG: {str(e)}")