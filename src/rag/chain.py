from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import streamlit as st
import logging
from datetime import datetime
from config import LLM_MODEL

logger = logging.getLogger(__name__)

def setup_rag_chain(retriever):
    """Chain con debugging esteso."""
    
    llm = ChatOpenAI(
        model_name=LLM_MODEL,
        temperature=0,
        api_key=st.secrets["OPENAI_API_KEY"]
    )
    
    def get_response(query_input):
        try:
            query = query_input.get("query", "") if isinstance(query_input, dict) else query_input
            
            # DEBUG: Stampa la query
            st.write("DEBUG - Query:", query)
            
            # Recupera i documenti
            docs = retriever.get_relevant_documents(query)
            
            # DEBUG: Stampa numero di documenti
            st.write(f"DEBUG - Documenti recuperati: {len(docs)}")
            
            if not docs:
                st.write("DEBUG - Nessun documento trovato!")
                return {"result": "Non ho trovato documenti rilevanti."}

            # DEBUG: Stampa dettagli di ogni documento
            st.write("\nDEBUG - Dettagli documenti:")
            for i, doc in enumerate(docs):
                st.write(f"\nDocumento {i+1}:")
                st.write(f"Thread ID: {doc.metadata.get('thread_id')}")
                st.write(f"Post Time: {doc.metadata.get('post_time')}")
                st.write(f"Total Posts: {doc.metadata.get('total_posts')}")
                st.write(f"Author: {doc.metadata.get('author')}")
                st.write("---")

            # Prepara il contesto per il modello
            context = "DOCUMENTI ANALIZZATI:\n\n"
            for doc in docs:
                context += f"""POST:
Autore: {doc.metadata.get('author')}
Data: {doc.metadata.get('post_time')}
Thread: {doc.metadata.get('thread_title')}
Thread ID: {doc.metadata.get('thread_id')}
Total Posts nel Thread: {doc.metadata.get('total_posts')}
---
Contenuto:
{doc.page_content}
==========\n\n"""

            # DEBUG: Stampa il contesto completo
            st.write("\nDEBUG - Contesto inviato al modello:")
            st.write(context[:500] + "..." if len(context) > 500 else context)

            prompt = f"""ANALIZZA QUESTI POSTS DI FORUM:

{context}

IMPORTANTE: Nel forum sopra, conta ed elenca:
1. Il numero ESATTO di threads (distingui per thread_id)
2. Il numero TOTALE di posts che vedi
3. Confronta con il numero 'total_posts' nei metadata

Poi rispondi alla domanda: {query}"""

            st.write("\nDEBUG - Invio al modello...")
            
            messages = [
                SystemMessage(content="Sei un analista di forum. Il tuo primo compito è SEMPRE contare e riportare il numero esatto di thread e post che vedi."),
                HumanMessage(content=prompt)
            ]
            
            response = llm.invoke(messages)
            
            # DEBUG: Stampa la risposta
            st.write("\nDEBUG - Risposta ricevuta dal modello:", response.content if response else "Nessuna risposta")
            
            return {"result": response.content} if response else {"result": "Errore nella generazione della risposta."}
            
        except Exception as e:
            logger.error(f"Errore nella chain: {str(e)}")
            st.write(f"DEBUG - Errore: {str(e)}")
            return {"result": f"Errore: {str(e)}"}
    
    return get_response