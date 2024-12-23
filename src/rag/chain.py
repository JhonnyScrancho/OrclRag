from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import streamlit as st
import logging

logger = logging.getLogger(__name__)

class ForumPromptManager:
    def __init__(self):
        self.system_prompt = """Sei un assistente specializzato nell'analisi di conversazioni di forum tech.
Utilizza sempre questi metadati per arricchire le tue risposte:

- sentiment: per capire il tono del messaggio (-1 a +1)
- keywords: per identificare i temi chiave
- post_time: per il contesto temporale
- metadata.content_length: per dare peso ai post più elaborati

REGOLE FONDAMENTALI:
1. Cita SEMPRE il timestamp e l'autore quando riporti un post
2. Usa il sentiment per interpretare il contesto emotivo
3. Considera le keywords per capire il focus della discussione
4. Dai più peso ai post più lunghi e articolati
5. Traccia la conversazione attraverso le citazioni ("X said: ...")

Se ti chiedono statistiche o trend:
- Usa il sentiment per tracciare l'evoluzione della discussione
- Raggruppa per keywords per identificare i temi ricorrenti
- Analizza la frequenza delle interazioni tra utenti
- Considera la lunghezza dei post per valutare la profondità della discussione

FORMAT RISPOSTA:
1. Citazione diretta [timestamp, autore, sentiment]
2. Insight basato sui metadati disponibili
3. Conclusione supportata dai dati"""

    def build_conversation_prompt(self, context: str, query: str) -> str:
        return f"""ANALISI RICHIESTA: {query}

CONTESTO FORUM:
{context}

Ricorda di:
1. Usare i metadati (sentiment, keywords, lunghezza) per arricchire l'analisi
2. Citare sempre timestamp e autore
3. Evidenziare il tono della conversazione usando il sentiment score"""

def setup_rag_chain(retriever):
    """Configura una chain RAG ottimizzata per l'analisi di forum."""
    prompt_manager = ForumPromptManager()
    
    llm = ChatOpenAI(
        model_name="gpt-4-turbo-preview",
        temperature=0.3,
        api_key=st.secrets["OPENAI_API_KEY"]
    )
    
    def get_response(query_input):
        try:
            query = query_input.get("query", "") if isinstance(query_input, dict) else query_input
            
            docs = retriever.get_all_documents()
            if not docs:
                return {"result": "Non ho trovato dati sufficienti per rispondere."}

            # Prepara il contesto arricchito con metadati
            posts_context = []
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
                posts_context.append(post)
            
            # Ordina per timestamp
            posts_context.sort(key=lambda x: x["time"])
            
            # Formatta il contesto con metadati
            thread_title = posts_context[0]["thread_title"]
            context = f"Thread: {thread_title}\n\n" + "\n\n".join([
                f"[{post['time']}] {post['author']}"
                f"\nSentiment: {post['sentiment']}"
                f"\nKeywords: {', '.join(post['keywords'])}"
                f"\nLength: {post['content_length']}"
                f"\nContent:\n{post['content']}"
                for post in posts_context
            ])
            
            # Costruisci il prompt finale
            conversation_prompt = prompt_manager.build_conversation_prompt(context, query)
            
            messages = [
                SystemMessage(content=prompt_manager.system_prompt),
                HumanMessage(content=conversation_prompt)
            ]
            
            response = llm.invoke(messages)
            return {"result": response.content}
            
        except Exception as e:
            logger.error(f"Error in RAG chain: {str(e)}")
            return {"result": f"Errore nell'elaborazione: {str(e)}"}
    
    return get_response