from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import streamlit as st
import logging
from datetime import datetime
from config import LLM_MODEL

logger = logging.getLogger(__name__)

class ForumMetadataManager:
    def __init__(self):
        self.system_prompt = """Sei un analista esperto di contenuti da forum. Il tuo compito è analizzare e fornire insight accurati basati sui thread del forum, considerando sempre tutti i posts disponibili. Assicurati di:

1. CONTEGGIO E ANALISI:
   - Conta sempre il numero esatto di threads e posts disponibili
   - Verifica la concordanza tra posts trovati e dichiarati
   - Analizza la cronologia completa della discussione

2. QUALITÀ DATI:
   - Utilizza tutti i metadati disponibili (date, autori, sentiment)
   - Verifica la completezza dei dati
   - Segnala eventuali incongruenze nei dati

3. RISPOSTA:
   - Inizia SEMPRE specificando il numero esatto di threads e posts analizzati
   - Fornisci il range temporale preciso della discussione
   - Indica il livello di confidenza basato sulla completezza dei dati
   - Procedi poi con l'analisi richiesta

4. FORMATO:
---
RISPOSTA DIRETTA:
[Indicare SEMPRE: "Sono stati analizzati X threads contenenti Y posts."]

LIVELLO DI CONFIDENZA:
[Alto/Medio/Basso] basato su:
- Numero fonti: [X threads, Y posts]
- Range temporale: [prima data - ultima data]
- Sentiment medio: [valore]

APPROFONDIMENTO:
[Analisi dettagliata richiesta]
---"""

    def build_conversation_prompt(self, context: str, query: str) -> str:
        # Estrai il conteggio dei threads e posts dal contesto
        thread_count = context.count("THREAD:")
        return f"""QUERY: {query}

ATTENZIONE: Il contesto contiene {thread_count} threads. Assicurati di analizzarli tutti.

CONTESTO FORUM:
{context}

RICHIESTE SPECIFICHE:
1. Conta e riporta SEMPRE il numero esatto di threads e posts
2. Verifica la concordanza tra il numero di posts trovati e dichiarati
3. Includi sempre il range temporale completo
4. Rispondi poi alla query specifica

FORMAT0 RISPOSTA RICHIESTO:
---
RISPOSTA DIRETTA:
Sono stati analizzati [X] threads contenenti [Y] posts.
[Risposta alla query specifica]

LIVELLO DI CONFIDENZA:
[Alto/Medio/Basso] basato su:
- Numero fonti: [X] threads, [Y] posts
- Range temporale: [prima data - ultima data]
- Sentiment medio: [valore]

APPROFONDIMENTO:
[Analisi dettagliata se richiesta]
---"""

def setup_rag_chain(retriever):
    """Configura RAG chain con gestione migliorata dei documenti."""
    prompt_manager = ForumMetadataManager()
    
    llm = ChatOpenAI(
        model_name=LLM_MODEL,
        temperature=0.3,
        api_key=st.secrets["OPENAI_API_KEY"]
    )
    
    def get_response(query_input):
        try:
            query = query_input.get("query", "") if isinstance(query_input, dict) else query_input
            
            # Recupera i documenti rilevanti
            relevant_docs = retriever.get_relevant_documents(query)
            if not relevant_docs:
                return {"result": "Mi dispiace, non ho trovato informazioni rilevanti per rispondere alla tua domanda."}

            # Struttura dati per l'analisi
            threads_data = {}
            all_timestamps = []
            all_sentiments = []
            
            # Primo passaggio: raggruppa per thread e raccolta dati
            for doc in relevant_docs:
                meta = doc.metadata
                thread_id = meta.get("thread_id", "unknown")
                
                if thread_id not in threads_data:
                    threads_data[thread_id] = {
                        "title": meta.get("thread_title", "Unknown Thread"),
                        "total_posts": meta.get("total_posts", 0),
                        "declared_posts": meta.get("declared_posts", 0),
                        "scrape_time": meta.get("scrape_time", "Unknown"),
                        "posts": []
                    }
                
                # Aggiungi il post ai dati del thread
                post_time = meta.get("post_time", "Unknown")
                sentiment = meta.get("sentiment", 0)
                
                threads_data[thread_id]["posts"].append({
                    "post_id": meta.get("post_id", "unknown"),
                    "author": meta.get("author", "Unknown"),
                    "time": post_time,
                    "content": doc.page_content,
                    "sentiment": sentiment,
                    "keywords": meta.get("keywords", [])
                })
                
                if post_time != "Unknown":
                    all_timestamps.append(post_time)
                if isinstance(sentiment, (int, float)):
                    all_sentiments.append(sentiment)
            
            # Prepara il contesto strutturato
            context = "STATISTICHE GLOBALI:\n"
            context += f"Threads analizzati: {len(threads_data)}\n"
            total_posts = sum(len(thread["posts"]) for thread in threads_data.values())
            total_declared = sum(thread["declared_posts"] for thread in threads_data.values())
            context += f"Posts trovati: {total_posts}\n"
            context += f"Posts dichiarati nei metadata: {total_declared}\n"
            
            if all_timestamps:
                context += f"Range temporale: {min(all_timestamps)} - {max(all_timestamps)}\n"
            
            avg_sentiment = sum(all_sentiments) / len(all_sentiments) if all_sentiments else 0
            context += f"Sentiment medio: {avg_sentiment:.2f}\n\n"
            
            # Aggiungi dettagli per ogni thread
            for thread_id, thread in threads_data.items():
                context += f"THREAD: {thread['title']}\n"
                context += f"Posts trovati: {len(thread['posts'])}\n"
                context += f"Posts dichiarati: {thread['declared_posts']}\n"
                
                # Ordina i posts per timestamp
                thread["posts"].sort(key=lambda x: x["time"])
                
                for post in thread["posts"]:
                    context += f"\n[{post['time']}] {post['author']}\n"
                    context += f"Sentiment: {post['sentiment']}\n"
                    context += f"Keywords: {', '.join(post['keywords'])}\n"
                    context += f"Content:\n{post['content']}\n"
            
            # Crea il prompt finale
            conversation_prompt = prompt_manager.build_conversation_prompt(context, query)
            
            messages = [
                SystemMessage(content=prompt_manager.system_prompt),
                HumanMessage(content=conversation_prompt)
            ]
            
            # Genera la risposta
            response = llm.invoke(messages)
            if not response or not hasattr(response, 'content'):
                return {"result": "Mi dispiace, ho avuto un problema nel generare una risposta."}
            
            return {"result": response.content}
            
        except Exception as e:
            logger.error(f"Errore nella RAG chain: {str(e)}")
            return {"result": f"Si è verificato un errore durante l'elaborazione della tua richiesta: {str(e)}"}
    
    return get_response