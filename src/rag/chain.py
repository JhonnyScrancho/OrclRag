from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
import streamlit as st
import json
import logging

logger = logging.getLogger(__name__)

def setup_rag_chain(retriever):
    """Configura una chain RAG semplificata che sfrutta le capacità di comprensione del LLM."""
    llm = ChatOpenAI(
        model_name="gpt-4-turbo-preview",
        temperature=0.3,
        api_key=st.secrets["OPENAI_API_KEY"]
    )
    
    template = """Sei un assistente esperto nell'analisi di conversazioni dei forum. Hai accesso ai dati di un thread del forum.
Nel rispondere, presta particolare attenzione a:
1. Identificare e utilizzare le citazioni presenti (formato "utente said: contenuto")
2. Comprendere il flusso della conversazione e chi risponde a chi
3. Interpretare correttamente il contesto temporale dei post
4. Evidenziare le citazioni rilevanti quando rispondi

Dati del forum:
{context}

Domanda: {query}

Fornisci una risposta concisa e pertinente in italiano, citando le parti rilevanti della conversazione quando appropriato.
Quando citi un post, usa il formato: "[Autore] ha scritto: '...'"""
    
    def get_response(query_input):
        try:
            # Gestisci sia input stringa che dizionario
            query = query_input.get("query", "") if isinstance(query_input, dict) else query_input
            
            # Ottieni i documenti rilevanti
            docs = retriever.get_all_documents()
            if not docs:
                return {"result": "Non ho trovato dati sufficienti per rispondere."}
            
            # Prepara il contesto come una sequenza temporale di post
            posts_context = []
            for doc in docs:
                post = {
                    "author": doc.metadata.get("author", "Unknown"),
                    "time": doc.metadata.get("post_time", "Unknown"),
                    "content": doc.metadata.get("text", ""),
                    "thread_title": doc.metadata.get("thread_title", "Unknown Thread")
                }
                posts_context.append(post)
            
            # Ordina i post per timestamp
            posts_context.sort(key=lambda x: x["time"])
            
            # Aggiungi il titolo del thread al contesto
            thread_title = posts_context[0]["thread_title"] if posts_context else "Unknown Thread"
            
            # Formatta il contesto come una conversazione
            context = f"Thread: {thread_title}\n\n" + "\n\n".join([
                f"[{post['time']}] {post['author']}:\n{post['content']}"
                for post in posts_context
            ])
            
            messages = [
                SystemMessage(content="Sei un assistente esperto nell'analisi di conversazioni dei forum."),
                HumanMessage(content=template.format(context=context, query=query))
            ]
            
            response = llm.invoke(messages)
            return {"result": response.content}
            
        except Exception as e:
            logger.error(f"Error in RAG chain: {str(e)}")
            return {"result": f"Errore nell'elaborazione: {str(e)}"}
    
    return get_response