from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import streamlit as st
import logging
from datetime import datetime
from config import LLM_MODEL

logger = logging.getLogger(__name__)

class ForumMetadataManager:
    def __init__(self):
        self.system_prompt = """Sei un analista esperto di contenuti da forum. Il tuo compito è interpretare e aggregare discussioni per fornire insights su qualsiasi argomento richiesto, sfruttando sia il contenuto che i metadati disponibili.

ANALISI DATI:
1. CONTESTUALE:
   - Identifica il tema principale della richiesta
   - Trova discussioni rilevanti
   - Valuta la pertinenza delle informazioni
   - Considera il contesto temporale (scrape_time vs post_time)

2. QUALITATIVA:
   - Consenso della community
   - Esperienze dirette vs indirette
   - Credibilità delle fonti
   - Peso dei contributi (content_length)
   - Evoluzione del sentiment nella discussione

3. CORRELAZIONI:
   - Pattern di keywords ricorrenti
   - Collegamenti tra discussioni
   - Flusso delle conversazioni (citazioni)
   - Interazioni tra utenti
   - Evoluzione dei temi nel tempo

4. METRICHE:
   - Engagement (total_posts, lunghezza risposte)
   - Sentiment trend (-1 a +1)
   - Densità keywords
   - Autorevolezza utenti (frequenza, qualità interventi)

RISPOSTE:
1. SINTESI:
   - Risposta diretta alla domanda
   - Evidenze dai thread più rilevanti
   - Supporto con metriche chiave
   - Grado di affidabilità basato sui metadati

2. APPROFONDIMENTO (se richiesto):
   - Analisi temporale dei trend
   - Pattern di sentiment e keywords
   - Citazioni rilevanti con contesto
   - Correlazioni tra discussioni
   - Implicazioni pratiche"""

    def build_conversation_prompt(self, context: str, query: str) -> str:
        return f"""QUERY: {query}

CONTESTO FORUM:
{context}

ISTRUZIONI:
1. Fornisci sempre prima una risposta diretta e concisa alla query
2. Se la query richiede approfondimenti, aggiungi l'analisi dettagliata dopo
3. Usa i metadati disponibili per supportare le tue conclusioni:
   - Sentiment per il tono delle discussioni
   - Keywords per i temi principali
   - Timestamps per l'attualità
   - Content length per il peso dei contributi
4. Evidenzia sempre il livello di confidenza basato su:
   - Numero di fonti concordanti
   - Attualità delle informazioni
   - Autorevolezza degli utenti
   - Qualità dei metadati disponibili

FORMATO RISPOSTA:
---
RISPOSTA DIRETTA:
[Risposta concisa alla query]

LIVELLO DI CONFIDENZA:
[Alto/Medio/Basso] basato su:
- Numero fonti: [X]
- Range temporale: [data più vecchia - data più recente]
- Sentiment medio: [valore]

APPROFONDIMENTO (se richiesto):
[Analisi dettagliata]
---"""

def setup_rag_chain(retriever):
    """Configura RAG chain con gestione errori migliorata."""
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

            # Strutture dati per l'analisi
            threads_data = {}  # Per raggruppare i post per thread
            all_sentiments = []
            all_timestamps = []
            
            # Prima passata: raggruppamento per thread e raccolta metadati
            for doc in relevant_docs:
                if not isinstance(doc.metadata, dict):
                    continue
                    
                thread_id = doc.metadata.get("thread_id", "unknown")
                if thread_id not in threads_data:
                    threads_data[thread_id] = {
                        "title": doc.metadata.get("thread_title", "Unknown Thread"),
                        "total_posts": doc.metadata.get("total_posts", 0),
                        "scrape_time": doc.metadata.get("scrape_time", "Unknown"),
                        "posts": []
                    }
                
                # Aggiungi il post ai dati del thread
                post_data = {
                    "post_id": doc.metadata.get("post_id", "unknown"),
                    "author": doc.metadata.get("author", "Unknown"),
                    "time": doc.metadata.get("post_time", "Unknown"),
                    "content": doc.page_content,
                    "sentiment": doc.metadata.get("sentiment", 0),
                    "keywords": doc.metadata.get("keywords", []),
                    "content_length": len(doc.page_content) if doc.page_content else 0
                }
                
                threads_data[thread_id]["posts"].append(post_data)
                
                # Raccogli dati per metriche
                if post_data["time"] != "Unknown":
                    all_timestamps.append(post_data["time"])
                if isinstance(post_data["sentiment"], (int, float)):
                    all_sentiments.append(post_data["sentiment"])

            # Calcolo metriche globali
            total_threads = len(threads_data)
            total_posts = sum(len(thread["posts"]) for thread in threads_data.values())
            thread_declared_posts = sum(thread.get("total_posts", 0) for thread in threads_data.values())
            avg_sentiment = sum(all_sentiments) / len(all_sentiments) if all_sentiments else 0
            time_range = f"{min(all_timestamps)} - {max(all_timestamps)}" if all_timestamps else "Unknown"

            # Formatta il contesto
            context = f"""STATISTICHE GLOBALI:
    Threads analizzati: {total_threads}
    Posts trovati: {total_posts}
    Posts dichiarati nei metadata: {thread_declared_posts}
    Range temporale: {time_range}
    Sentiment medio: {avg_sentiment:.2f}

    DETTAGLIO THREADS:"""

            # Aggiungi dettagli per ogni thread
            for thread_id, thread_data in threads_data.items():
                context += f"\n\nTHREAD: {thread_data['title']}"
                context += f"\nTotal posts: {thread_data['total_posts']}"
                context += f"\nScrape time: {thread_data['scrape_time']}"
                context += "\n\nPOSTS:"
                
                # Ordina i post per timestamp
                thread_data["posts"].sort(key=lambda x: x["time"])
                
                for post in thread_data["posts"]:
                    context += f"\n\n[{post['time']}] {post['author']}"
                    context += f"\nSentiment: {post['sentiment']}"
                    context += f"\nKeywords: {', '.join(post['keywords'])}"
                    context += f"\nLunghezza: {post['content_length']}"
                    context += f"\nContenuto:\n{post['content']}"

            # Preparazione del prompt
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