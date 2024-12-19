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
    
    template = """Sei un assistente esperto che analizza e risponde a domande su thread di forum. Analizza attentamente il contesto fornito e rispondi in modo approfondito.

    Per le statistiche (numero thread/post/citazioni):
    - Usa i dati presenti nel contesto
    - Includi dettagli aggiuntivi rilevanti se disponibili
    
    Per l'analisi dei thread:
    - Identifica l'argomento principale
    - Riassumi i punti chiave della discussione
    - Evidenzia le interazioni principali tra gli utenti
    - Menziona informazioni rilevanti come sentiment, keywords ricorrenti
    
    Per le citazioni:
    - Identifica quando un utente cita un altro (pattern: "User said: ... Click to expand...")
    - Traccia la conversazione e il contesto delle citazioni
    
    Contesto fornito:
    {context}
    
    Domanda: {query}
    
    Rispondi in modo esaustivo e naturale, come faresti in una normale conversazione. Se davvero non trovi informazioni pertinenti nel contesto, rispondi "Mi dispiace, non ho trovato informazioni sufficienti per rispondere alla tua domanda."
    
    Risposta in italiano:"""
    
    prompt = PromptTemplate(template=template, input_variables=["context", "query"])
    
    def get_response(query_input):
        try:
            if isinstance(query_input, dict):
                query = query_input.get("query", "")
            else:
                query = query_input
            
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join(doc.page_content for doc in docs)
            
            if not context.strip():
                return {"result": "Mi dispiace, non ho trovato informazioni sufficienti per rispondere alla tua domanda."}
            
            formatted_prompt = prompt.format(context=context, query=query)
            messages = [HumanMessage(content=formatted_prompt)]
            response = llm.invoke(messages)
            
            result = response.content if hasattr(response, 'content') else str(response)
            return {"result": result}
            
        except Exception as e:
            st.error(f"Errore nella catena RAG: {str(e)}")
            return {"result": "Mi dispiace, c'è stato un errore nell'elaborazione della risposta."}
    
    return get_response