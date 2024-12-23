from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import streamlit as st
import logging
from datetime import datetime
from config import LLM_MODEL

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
            
            if not posts_data:
                return {"result": "Ho trovato dei documenti ma non sono riuscito a processarli correttamente."}
            
            # Ordina i post per timestamp se possibile
            try:
                posts_data.sort(key=lambda x: x["time"])
            except Exception as e:
                logger.warning(f"Impossibile ordinare i post per timestamp: {e}")
            
            # Formatta il contesto con focus sui metadati
            context = "Thread: " + posts_data[0]['thread_title'] + "\n\n"
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