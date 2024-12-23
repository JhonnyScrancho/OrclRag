from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import streamlit as st
import logging
from datetime import datetime
from config import LLM_MODEL

logger = logging.getLogger(__name__)

def setup_rag_chain(retriever):
    """Configura una RAG chain semplificata."""
    
    system_prompt = """Sei un assistente che analizza discussioni di forum.
Il tuo primo compito è SEMPRE contare e riportare il numero esatto di thread e post.
Inizia SEMPRE la tua risposta con il conteggio preciso."""

    llm = ChatOpenAI(
        model_name=LLM_MODEL,
        temperature=0,  # Ridotto a 0 per massima precisione
        api_key=st.secrets["OPENAI_API_KEY"]
    )
    
    def get_response(query_input):
        try:
            query = query_input.get("query", "") if isinstance(query_input, dict) else query_input
            
            # Recupera i documenti
            docs = retriever.get_relevant_documents(query)
            if not docs:
                return {"result": "Non ho trovato documenti rilevanti."}

            # Conta thread e post
            thread_ids = set()
            post_count = 0
            dates = []
            sentiments = []
            
            for doc in docs:
                thread_id = doc.metadata.get("thread_id")
                if thread_id:
                    thread_ids.add(thread_id)
                    post_count += 1
                    
                    post_time = doc.metadata.get("post_time")
                    if post_time:
                        dates.append(post_time)
                        
                    sentiment = doc.metadata.get("sentiment")
                    if sentiment is not None:
                        sentiments.append(sentiment)

            # Prepara il contesto
            context = f"""STATISTICHE:
Thread trovati: {len(thread_ids)}
Post totali: {post_count}
Range date: {min(dates) if dates else 'N/A'} - {max(dates) if dates else 'N/A'}
Sentiment medio: {sum(sentiments)/len(sentiments) if sentiments else 'N/A'}

CONTENUTO:
"""

            # Aggiungi il contenuto dei documenti
            for doc in docs:
                context += f"\nPOST:\n{doc.page_content}\n"
                for key, value in doc.metadata.items():
                    if key in ['author', 'post_time', 'sentiment']:
                        context += f"{key}: {value}\n"
                context += "---\n"

            # Crea il prompt
            prompt = f"""QUERY: {query}

{context}

RICORDA: Inizia SEMPRE specificando il numero esatto di thread e post trovati.
Poi procedi con la risposta alla domanda."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ]
            
            response = llm.invoke(messages)
            return {"result": response.content} if response else {"result": "Errore nella generazione della risposta."}
            
        except Exception as e:
            logger.error(f"Errore nella chain: {str(e)}")
            return {"result": f"Errore: {str(e)}"}
    
    return get_response