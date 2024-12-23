from langchain_core.messages import HumanMessage, SystemMessage
import streamlit as st
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ForumMetadataManager:
    def __init__(self):
        self.system_prompt = """Sei un analista esperto nell'analisi di conversazioni di forum.
Utilizza sempre questi metadati per arricchire le tue risposte:

METADATI DISPONIBILI PER OGNI POST:
1. SENTIMENT: Score da -1 a +1 che indica il tono emotivo
2. KEYWORDS: Parole chiave estratte automaticamente
3. POST_TIME: Timestamp del post per analisi temporale
4. CONTENT_LENGTH: Lunghezza del post come indicatore di dettaglio

CITAZIONI:
Il contenuto può contenere citazioni nel formato:
"[Autore] said: [testo citato] Click to expand... [risposta]"
- Identifica chi risponde a chi
- Analizza il contesto delle risposte
- Traccia le conversazioni tra utenti
- Valuta il tono delle risposte usando il sentiment

FORMATO RISPOSTA:
1. [Timestamp, Autore] Citazione rilevante
   - Sentiment: {valore}
   - Keywords chiave: [lista]
   - Lunghezza: {valore}
   - Citazioni: menziona se il post è una risposta
2. Analisi basata sui metadati disponibili
3. Pattern di interazione identificati
4. Conclusione supportata dai dati"""

    def build_conversation_prompt(self, context: str, query: str) -> str:
        return f"""QUERY: {query}

CONTESTO FORUM:
{context}

LINEE GUIDA:
1. Usa SEMPRE i metadati disponibili per contestualizzare
2. Cita con timestamp e autore
3. Evidenzia sentiment significativi
4. Identifica pattern nelle keywords
5. Considera la lunghezza dei post per il peso
6. Analizza le interazioni attraverso le citazioni
7. Traccia il filo delle conversazioni basandoti sul formato delle citazioni"""

def setup_rag_chain(retriever):
    """Configura RAG chain focalizzata sui metadati."""
    prompt_manager = ForumMetadataManager()
    
    def get_response(query_input):
        try:
            query = query_input.get("query", "") if isinstance(query_input, dict) else query_input
            
            docs = retriever.get_all_documents()
            if not docs:
                return {"result": "Dati insufficienti per l'analisi."}

            # Estrai e organizza i metadati
            posts_data = []
            for doc in docs:
                metadata = doc.metadata
                post = {
                    "author": metadata.get("author", "Unknown"),
                    "time": metadata.get("post_time", "Unknown"),
                    "content": metadata.get("text", ""),
                    "sentiment": metadata.get("sentiment", 0),
                    "keywords": metadata.get("keywords", []),
                    "content_length": metadata.get("content_length", 0),
                    "thread_title": metadata.get("thread_title", "Unknown Thread")
                }
                posts_data.append(post)
            
            # Ordina per timestamp
            posts_data.sort(key=lambda x: x["time"])
            
            # Formatta il contesto con focus sui metadati
            context = f"Thread: {posts_data[0]['thread_title']}\n\n" + "\n\n".join([
                f"[{post['time']}] {post['author']}"
                f"\nSentiment: {post['sentiment']}"
                f"\nKeywords: {', '.join(post['keywords'])}"
                f"\nLunghezza: {post['content_length']}"
                f"\nContenuto:\n{post['content']}"
                for post in posts_data
            ])
            
            conversation_prompt = prompt_manager.build_conversation_prompt(context, query)
            
            messages = [
                SystemMessage(content=prompt_manager.system_prompt),
                HumanMessage(content=conversation_prompt)
            ]
            
            return {"result": messages}
            
        except Exception as e:
            logger.error(f"Error in RAG chain: {str(e)}")
            return {"result": f"Errore nell'elaborazione: {str(e)}"}
    
    return get_response