from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from config import LLM_MODEL
import streamlit as st

def setup_rag_chain(retriever):
    llm = ChatOpenAI(
        model_name=LLM_MODEL,
        temperature=0,
        api_key=st.secrets["OPENAI_API_KEY"]
    )
    
    template = """Sei un assistente che analizza dati di forum. Fornisci risposte precise e concise, senza aggiungere informazioni non richieste.

Se la domanda riguarda statistiche (numero di thread/post):
- Riporta solo i numeri esatti
- Non aggiungere interpretazioni o analisi non richieste

Se la domanda riguarda citazioni:
- Riporta solo il numero di citazioni trovate
- Se richiesto, elenca le citazioni specifiche

Se la domanda riguarda il contenuto/riassunto:
- Descrivi l'argomento principale
- Elenca i punti chiave della discussione
- Mantieni il focus sul contenuto effettivo

Contesto fornito:
{context}

Domanda: {query}

Rispondi in modo diretto e conciso in italiano."""
    
    prompt = PromptTemplate(template=template, input_variables=["context", "query"])
    
    def get_response(query_input):
        try:
            query = query_input.get("query", "") if isinstance(query_input, dict) else query_input
            docs = retriever.get_relevant_documents(query)
            
            if not docs:
                return {"result": "Mi dispiace, non ho trovato informazioni sufficienti per rispondere alla tua domanda."}
            
            context = "\n\n".join(doc.page_content for doc in docs)
            messages = [HumanMessage(content=prompt.format(context=context, query=query))]
            response = llm.invoke(messages)
            
            return {"result": response.content if hasattr(response, 'content') else str(response)}
            
        except Exception as e:
            st.error(f"Errore nella catena RAG: {str(e)}")
            return {"result": "Mi dispiace, c'è stato un errore nell'elaborazione della risposta."}
    
    return get_response