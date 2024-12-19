from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
import streamlit as st
import json
import logging

logger = logging.getLogger(__name__)

def setup_rag_chain(retriever):
    """Configura una chain RAG più diretta e precisa."""
    llm = ChatOpenAI(
        model_name="gpt-4-turbo-preview",
        temperature=0.3,  # Ridotta per risposte più precise
        api_key=st.secrets["OPENAI_API_KEY"]
    )
    
    template = """Sei un assistente diretto e preciso. Rispondi alle domande in modo conciso utilizzando i dati forniti.

REGOLE:
1. Rispondi SOLO a ciò che viene chiesto
2. Sii breve e diretto
3. Per domande numeriche, dai prima il numero e poi solo insight essenziali
4. Se rilevi citazioni, indicale esplicitamente
5. Non fare analisi non richieste

Dati del forum:
{context}

Domanda: {query}

Fornisci una risposta concisa e pertinente in italiano."""
    
    def get_response(query_input):
        try:
            query = query_input.get("query", "") if isinstance(query_input, dict) else query_input
            docs = retriever.get_relevant_documents(query)
            
            if not docs:
                return {"result": "Non ho trovato dati sufficienti per rispondere."}
            
            rich_analysis = docs[0].metadata.get("analysis", {})
            if not rich_analysis:
                return {"result": "Non ho accesso ai dati necessari."}
            
            context = json.dumps(rich_analysis, indent=2, ensure_ascii=False)
            
            messages = [
                SystemMessage(content="Sei un assistente preciso e conciso. Rispondi solo a ciò che viene chiesto."),
                HumanMessage(content=template.format(context=context, query=query))
            ]
            
            response = llm.invoke(messages)
            return {"result": response.content}
            
        except Exception as e:
            logger.error(f"Error in RAG chain: {str(e)}")
            return {"result": f"Errore nell'elaborazione: {str(e)}"}
    
    return get_response