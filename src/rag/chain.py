import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from config import LLM_MODEL

def setup_rag_chain(retriever):
    llm = ChatOpenAI(
        model_name=LLM_MODEL,
        temperature=0,
        api_key=st.secrets["OPENAI_API_KEY"]
    )
    
    template = """Sei un assistente esperto nell'analisi di thread di forum. Analizza attentamente il contesto fornito e rispondi in modo dettagliato e naturale.

Per le statistiche:
- Fornisci i numeri precisi di thread e post
- Aggiungi dettagli rilevanti sul contesto

Per le citazioni:
- Identifica le citazioni nel formato "User said: ... Click to expand..."
- Spiega il contesto delle citazioni

Per i riassunti:
- Identifica il tema principale
- Evidenzia i punti chiave della discussione
- Descrivi le interazioni principali tra gli utenti
- Includi dettagli su sentiment e argomenti ricorrenti

Contesto fornito:
{context}

Domanda: {query}

Rispondi in modo completo e naturale, come in una conversazione reale. Se non trovi le informazioni necessarie, spiega cosa manca."""
    
    prompt = PromptTemplate(template=template, input_variables=["context", "query"])
    
    def get_response(query_input):
        try:
            query = query_input.get("query", "") if isinstance(query_input, dict) else query_input
            docs = retriever.get_relevant_documents(query)
            
            if not docs:
                return {"result": "Mi dispiace, non ho trovato informazioni sufficienti nel database per rispondere alla tua domanda."}
            
            context = "\n\n".join(doc.page_content for doc in docs)
            messages = [HumanMessage(content=prompt.format(context=context, query=query))]
            response = llm.invoke(messages)
            
            return {"result": response.content if hasattr(response, 'content') else str(response)}
            
        except Exception as e:
            st.error(f"Errore nella catena RAG: {str(e)}")
            return {"result": "Mi dispiace, c'è stato un errore nell'elaborazione della risposta."}
    
    return get_response