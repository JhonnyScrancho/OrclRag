from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
import streamlit as st
import json
import logging
from datetime import datetime

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
Quando citi un post, usa il formato: "[Autore] ha scritto: '...'

REGOLE:
1. Rispondi SOLO a ciò che viene chiesto
2. Sii breve e diretto
3. Per domande numeriche, dai prima il numero e poi solo insight essenziali
4. Se rilevi citazioni, indicale esplicitamente
5. Non fare analisi non richieste"""
    
    def get_response(query_input):
        try:
            # Gestisci sia input stringa che dizionario
            query = query_input.get("query", "") if isinstance(query_input, dict) else query_input
            
            # Log della query ricevuta
            logger.info(f"Processing query: {query}")
            
            # Ottieni i documenti rilevanti
            docs = retriever.get_all_documents()
            logger.info(f"Retrieved {len(docs)} documents from the database")
            
            if not docs:
                logger.warning("No documents found in database")
                return {"result": "Non ho trovato dati sufficienti per rispondere."}
            
            # Log dettagliato dei documenti recuperati
            logger.info("Document details:")
            for i, doc in enumerate(docs):
                logger.info(f"Doc {i+1}: Author: {doc.metadata.get('author')}, Time: {doc.metadata.get('post_time')}")
            
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
            try:
                posts_context.sort(key=lambda x: datetime.fromisoformat(x["time"]))
                logger.info("Posts sorted by timestamp successfully")
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not sort posts by timestamp: {e}")
            
            # Aggiungi il titolo del thread al contesto
            thread_title = posts_context[0]["thread_title"] if posts_context else "Unknown Thread"
            
            # Formatta il contesto come una conversazione
            context = f"Thread: {thread_title}\n\n" + "\n\n".join([
                f"[{post['time']}] {post['author']}:\n{post['content']}"
                for post in posts_context
            ])
            
            logger.info(f"Prepared context with {len(posts_context)} posts")
            
            # Costruisci i messaggi per il LLM
            messages = [
                SystemMessage(content="Sei un assistente esperto nell'analisi di conversazioni dei forum."),
                HumanMessage(content=template.format(context=context, query=query))
            ]
            
            # Ottieni la risposta dal LLM
            try:
                response = llm.invoke(messages)
                logger.info("LLM response received successfully")
                return {"result": response.content}
            except Exception as e:
                logger.error(f"Error getting LLM response: {str(e)}")
                return {"result": f"Errore nella generazione della risposta: {str(e)}"}
            
        except Exception as e:
            logger.error(f"Error in RAG chain: {str(e)}")
            return {"result": f"Errore nell'elaborazione: {str(e)}"}
    
    return get_response