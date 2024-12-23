from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import streamlit as st
import logging
from datetime import datetime
from config import LLM_MODEL, MAX_DOCUMENTS_PER_QUERY
from data.batch_processor import BatchDocumentProcessor

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
    """Configura RAG chain con gestione migliorata dei documenti e batch processing."""
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
            
            # Set per tracciare post univoci
            unique_posts = set()
            post_contents = set()  # Per evitare duplicati di contenuto
            
            # Prima passata: organizziamo i dati per thread
            for doc in relevant_docs:
                meta = doc.metadata
                thread_id = meta.get("thread_id", "unknown")
                post_id = meta.get("post_id", "unknown")
                content = doc.page_content.strip()
                
                # Skippa duplicati
                if post_id in unique_posts or content in post_contents:
                    continue
                    
                unique_posts.add(post_id)
                post_contents.add(content)
                
                if thread_id not in threads_data:
                    threads_data[thread_id] = {
                        "title": meta.get("thread_title", "Unknown Thread"),
                        "total_posts": meta.get("total_posts", 0),
                        "declared_posts": meta.get("declared_posts", 0),
                        "scrape_time": meta.get("scrape_time", "Unknown"),
                        "posts": [],
                        "timestamps": [],
                        "sentiments": []
                    }
                
                # Aggiungi il post
                post_time = meta.get("post_time", "Unknown")
                sentiment = meta.get("sentiment", 0)
                
                threads_data[thread_id]["posts"].append({
                    "post_id": post_id,
                    "author": meta.get("author", "Unknown"),
                    "time": post_time,
                    "content": content,
                    "sentiment": sentiment,
                    "keywords": meta.get("keywords", [])
                })
                
                if post_time != "Unknown":
                    threads_data[thread_id]["timestamps"].append(post_time)
                if isinstance(sentiment, (int, float)):
                    threads_data[thread_id]["sentiments"].append(sentiment)
            
            # Seconda passata: costruiamo il contesto
            context = []
            total_posts = 0
            all_timestamps = []
            all_sentiments = []
            
            for thread_id, thread in threads_data.items():
                # Ordina i post per timestamp
                thread["posts"].sort(key=lambda x: x["time"])
                
                # Aggiorna contatori globali
                total_posts += len(thread["posts"])
                all_timestamps.extend(thread["timestamps"])
                all_sentiments.extend(thread["sentiments"])
                
                # Aggiungi info thread
                thread_info = [
                    f"\nTHREAD: {thread['title']}",
                    f"Posts trovati: {len(thread['posts'])}",
                    f"Posts dichiarati: {thread['declared_posts']}"
                ]
                context.extend(thread_info)
                
                # Aggiungi posts
                for post in thread["posts"]:
                    post_info = [
                        f"\n[{post['time']}] {post['author']}",
                        f"Sentiment: {post['sentiment']}",
                        f"Keywords: {', '.join(post['keywords'])}",
                        f"Content:\n{post['content']}\n"
                    ]
                    context.extend(post_info)
            
            # Calcola statistiche globali
            avg_sentiment = sum(all_sentiments) / len(all_sentiments) if all_sentiments else 0
            time_range = f"{min(all_timestamps)} - {max(all_timestamps)}" if all_timestamps else "Unknown"
            
            # Prepara il contesto finale
            final_context = [
                "STATISTICHE GLOBALI:",
                f"Threads analizzati: {len(threads_data)}",
                f"Posts trovati: {total_posts}",
                f"Range temporale: {time_range}",
                f"Sentiment medio: {avg_sentiment:.2f}\n"
            ]
            final_context.extend(context)
            
            # Crea il prompt finale
            conversation_prompt = prompt_manager.build_conversation_prompt(
                "\n".join(final_context), 
                query
            )
            
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