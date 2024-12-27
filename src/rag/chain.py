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
        self.system_prompt = """Sei un assistente esperto nell'analisi di contenuti da forum.
Il tuo compito è fornire risposte utili basate sui thread del forum disponibili.
Se trovi informazioni parziali o incomplete, cerca comunque di fornire la migliore risposta possibile,
specificando eventuali limitazioni nelle informazioni disponibili.

Quando rispondi:
1. Usa tutte le informazioni disponibili, anche se parziali
2. Specifica sempre il livello di confidenza nella risposta
3. Se necessario, chiedi chiarimenti
4. Suggerisci alternative se non puoi rispondere direttamente
"""

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
        """
        Process a query and generate a response using RAG.
        Args:
            query_input (Union[str, Dict]): The input query string or dict with query key
        Returns:
            Dict: Response containing the generated result
        """
        try:
            # Normalize query input
            query = query_input.get("query", "") if isinstance(query_input, dict) else query_input
            logger.info(f"Processing query: {query}")

            # Get relevant documents
            relevant_docs = retriever.get_relevant_documents(query)
            
            if not relevant_docs:
                logger.warning("No relevant documents found for query")
                return {
                    "result": "Mi dispiace, ma non ho trovato informazioni sufficienti per rispondere alla tua domanda. "
                            "Potresti riformulare la domanda in modo diverso o fornire più contesto?"
                }

            # Organize documents by thread
            threads_data = {}
            unique_posts = set()
            post_contents = set()

            # First pass: organize data by thread
            for doc in relevant_docs:
                meta = doc.metadata
                thread_id = meta.get("thread_id", "unknown")
                post_id = meta.get("post_id", "unknown")
                content = doc.page_content.strip()

                # Skip duplicates
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

                # Add post data
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

            # Build context
            context_parts = ["STATISTICHE GLOBALI:"]
            total_posts = sum(len(thread["posts"]) for thread in threads_data.values())
            all_timestamps = []
            all_sentiments = []

            for thread_id, thread in threads_data.items():
                # Sort posts chronologically
                thread["posts"].sort(key=lambda x: x["time"])
                
                # Update global counters
                all_timestamps.extend(thread["timestamps"])
                all_sentiments.extend(thread["sentiments"])

                # Add thread info
                context_parts.extend([
                    f"\nTHREAD: {thread['title']}",
                    f"Posts trovati: {len(thread['posts'])}",
                    f"Posts dichiarati: {thread['declared_posts']}"
                ])

                # Add posts
                for post in thread["posts"]:
                    context_parts.extend([
                        f"\n[{post['time']}] {post['author']}",
                        f"Sentiment: {post['sentiment']}",
                        f"Keywords: {', '.join(post['keywords'])}",
                        f"Content:\n{post['content']}\n"
                    ])

            # Calculate global statistics
            stats = {
                "threads": len(threads_data),
                "total_posts": total_posts,
                "time_range": (
                    f"{min(all_timestamps)} - {max(all_timestamps)}" 
                    if all_timestamps else "Unknown"
                ),
                "avg_sentiment": (
                    sum(all_sentiments) / len(all_sentiments) 
                    if all_sentiments else 0
                )
            }

            # Add statistics to context
            context_parts.insert(1, f"""
    Threads analizzati: {stats['threads']}
    Posts trovati: {stats['total_posts']}
    Range temporale: {stats['time_range']}
    Sentiment medio: {stats['avg_sentiment']:.2f}
    """)

            # Build final context and prompt
            final_context = "\n".join(context_parts)
            conversation_prompt = prompt_manager.build_conversation_prompt(
                final_context, 
                query
            )

            # Generate response
            messages = [
                SystemMessage(content=prompt_manager.system_prompt),
                HumanMessage(content=conversation_prompt)
            ]

            response = llm.invoke(messages)
            if not response or not hasattr(response, 'content'):
                logger.error("Failed to generate response")
                return {
                    "result": "Mi dispiace, ho avuto un problema nel generare una risposta. "
                            "Per favore, riprova tra qualche istante."
                }

            logger.info("Successfully generated response")
            return {"result": response.content}

        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}")
            return {
                "result": f"Si è verificato un errore durante l'elaborazione della tua richiesta. "
                        f"Per favore, riprova o riformula la domanda in modo diverso."
            }
    return get_response