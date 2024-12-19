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
    
    template = """Sei un assistente italiano esperto che aiuta a consultare un database di posts di un forum. 
    Il contesto fornito include sempre statistiche del database e un riepilogo dei thread disponibili nella prima parte.

    Per domande sul contenuto del database:
    - Se chiedono statistiche generali, usa le informazioni dal riepilogo
    - Se chiedono di cosa parlano i thread, elenca i titoli dei thread disponibili
    - Se chiedono dettagli specifici, cerca nelle informazioni dei post
    
    Se davvero non trovi informazioni pertinenti, rispondi "Mi dispiace, non ho trovato informazioni rilevanti per rispondere alla tua domanda."

    Contesto fornito: {context}
    
    Domanda: {query}
    
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
                return {"result": "Mi dispiace, non ho trovato informazioni rilevanti per rispondere alla tua domanda."}
            
            formatted_prompt = prompt.format(context=context, query=query)
            messages = [HumanMessage(content=formatted_prompt)]
            response = llm.invoke(messages)
            
            result = response.content if hasattr(response, 'content') else str(response)
            return {"result": result}
            
        except Exception as e:
            st.error(f"Errore nella catena RAG: {str(e)}")
            return {"result": "Mi dispiace, c'è stato un errore nell'elaborazione della risposta."}
    
    return get_response