from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import streamlit as st
import logging
from datetime import datetime
from .swarm import OpenAISwarm
from .templates import template
import asyncio

logger = logging.getLogger(__name__)

def setup_rag_chain(retriever):
    """Configura una chain RAG con sistema multi-agente."""
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-16k",
        temperature=0.3,
        api_key=st.secrets["OPENAI_API_KEY"]
    )

    swarm = OpenAISwarm()
    
    def get_response(query_input):
        try:
            # Gestisci sia input stringa che dizionario
            query = query_input.get("query", "") if isinstance(query_input, dict) else query_input
            
            with st.status("🤖 Analizzando i thread...", expanded=True) as status:
                # Log della query ricevuta
                st.write("📝 Elaborazione query in corso...")
                logger.info(f"Processing query: {query}")
                
                # Ottieni i documenti rilevanti
                docs = retriever.get_relevant_documents(query)
                if not docs:
                    logger.warning("No documents found in database")
                    return {"result": "Non ho trovato dati sufficienti per rispondere."}
                
                num_docs = len(docs)
                st.write(f"📚 Recuperati {num_docs} documenti dal database")
                logger.info(f"Retrieved {num_docs} documents from the database")
                
                # Log dettagliato dei documenti recuperati
                logger.info("Document details:")
                for i, doc in enumerate(docs):
                    logger.info(f"Doc {i+1}: Author: {doc.metadata.get('author')}, Time: {doc.metadata.get('post_time')}")
                
                try:
                    # Inizializza event loop per lo swarm
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    # Processa i documenti con lo swarm
                    num_agents = st.session_state.get('num_agents', 3)
                    st.write(f"🤖 Avvio elaborazione con {num_agents} agenti...")
                    result = loop.run_until_complete(swarm.process_documents(docs, query, status))
                    
                    if not result:
                        raise ValueError("Empty result from multi-agent processing")
                        
                    status.update(label="✅ Analisi completata!", state="complete")
                    return {"result": result}

                except Exception as e:
                    # Se il processing multi-agente fallisce, usa il metodo standard
                    logger.warning(f"Multi-agent processing failed: {str(e)}. Falling back to standard processing.")
                    st.write("⚠️ Elaborazione multi-agente non disponibile, utilizzo metodo standard...")
                    
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
                    
                    # Costruisci i messaggi per il LLM
                    messages = [
                        SystemMessage(content="Sei un assistente esperto nell'analisi di conversazioni dei forum."),
                        HumanMessage(content=template.format(context=context, query=query))
                    ]
                    
                    # Ottieni la risposta dal LLM
                    try:
                        st.write("🤔 Elaborazione risposta standard...")
                        response = llm.invoke(messages)
                        if not response or not hasattr(response, 'content'):
                            raise ValueError("Invalid response from LLM")
                            
                        logger.info("LLM response received successfully")
                        status.update(label="✅ Analisi completata!", state="complete")
                        return {"result": response.content}
                    except Exception as e:
                        logger.error(f"Error getting LLM response: {str(e)}")
                        status.update(label="❌ Errore nell'analisi", state="error")
                        return {"result": f"Errore nella generazione della risposta: {str(e)}"}

        except Exception as e:
            logger.error(f"Error in RAG chain: {str(e)}")
            return {"result": f"Errore nell'elaborazione: {str(e)}"}

    return get_response