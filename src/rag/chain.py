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

            # Estrai e organizza i metadati
            posts_data = []
            timestamps = []
            all_keywords = []
            sentiments = []
            
            for doc in relevant_docs:
                if not isinstance(doc.metadata, dict):
                    continue
                    
                post = {
                    "author": doc.metadata.get("author", "Unknown"),
                    "time": doc.metadata.get("post_time", "Unknown"),
                    "content": doc.page_content,
                    "sentiment": doc.metadata.get("sentiment", 0),
                    "keywords": doc.metadata.get("keywords", []),
                    "content_length": len(doc.page_content) if doc.page_content else 0,
                    "thread_title": doc.metadata.get("thread_title", "Unknown Thread")
                }
                posts_data.append(post)
                
                # Raccogli dati per metriche
                if post["time"] != "Unknown":
                    timestamps.append(post["time"])
                sentiments.append(post["sentiment"])
                all_keywords.extend(post["keywords"])
            
            if not posts_data:
                return {"result": "Ho trovato dei documenti ma non sono riuscito a processarli correttamente."}
            
            # Ordina i post per timestamp
            try:
                posts_data.sort(key=lambda x: x["time"])
            except Exception as e:
                logger.warning(f"Impossibile ordinare i post per timestamp: {e}")
            
            # Calcola metriche
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            time_range = f"{min(timestamps)} - {max(timestamps)}" if timestamps else "Unknown"
            
            # Formatta il contesto con focus sui metadati
            context = f"Thread: {posts_data[0]['thread_title']}\n"
            context += f"Periodo: {time_range}\n"
            context += f"Sentiment medio: {avg_sentiment:.2f}\n"
            context += f"Numero totale post: {len(posts_data)}\n\n"
            
            context += "\n\n".join([
                f"[{post['time']}] {post['author']}"
                f"\nSentiment: {post['sentiment']}"
                f"\nKeywords: {', '.join(post['keywords']) if isinstance(post['keywords'], list) else ''}"
                f"\nLunghezza: {post['content_length']}"
                f"\nContenuto:\n{post['content']}"
                for post in posts_data
            ])
            
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