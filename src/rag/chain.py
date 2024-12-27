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
        self.system_prompt = """Sei un analista esperto di forum specializzato nell'identificare trend, pattern e correlazioni tra discussioni correlate.
Il tuo compito è analizzare approfonditamente thread multipli su un argomento specifico per fornire una comprensione completa del tema.

CAPACITÀ CHIAVE:
1. Analisi Multi-Thread
   - Collega informazioni tra thread diversi
   - Identifica opinioni ricorrenti e contrasti
   - Traccia l'evoluzione delle discussioni nel tempo

2. Comprensione Contestuale
   - Analizza le citazioni per capire il flusso delle conversazioni
   - Identifica gli utenti chiave e le loro prospettive
   - Valuta il sentiment generale e come cambia nel tempo

3. Sintesi e Pattern
   - Evidenzia trend emergenti
   - Identifica problemi ricorrenti
   - Collega cause ed effetti tra discussioni diverse

4. Analisi Temporale
   - Traccia come le opinioni evolvono nel tempo
   - Identifica cambiamenti significativi
   - Correla eventi temporali tra thread diversi

FOCUS SPECIFICI:
- Cerca connessioni nascoste tra thread apparentemente separati
- Identifica consensus e disaccordi tra gli utenti
- Evidenzia come le esperienze si ripetono o differiscono
- Analizza il contesto completo prima di trarre conclusioni

APPROCCIO ALLE RISPOSTE:
1. Fornisci una visione d'insieme dell'argomento
2. Evidenzia pattern e trend significativi
3. Supporta le conclusioni con esempi specifici dai thread
4. Segnala eventuali limitazioni nei dati
5. Suggerisci possibili correlazioni da investigare ulteriormente"""

    def build_conversation_prompt(self, context: str, query: str) -> str:
        thread_stats = self._extract_thread_stats(context)
        
        return f"""QUERY: {query}

CONTESTO FORUM:
{context}

LINEE GUIDA PER L'ANALISI:
1. Panoramica Generale
   - Trend principali identificati
   - Pattern ricorrenti
   - Evoluzione temporale

2. Analisi delle Correlazioni
   - Collegamenti tra thread
   - Citazioni significative
   - Opinioni contrastanti

3. Insight Chiave
   - Problemi ricorrenti
   - Soluzioni proposte
   - Esperienze comuni

4. Contesto Temporale
   - Cambiamenti nel tempo
   - Eventi significativi
   - Sviluppi recenti

La tua risposta deve:
- Sintetizzare informazioni da tutti i thread pertinenti
- Evidenziare le connessioni più significative
- Supportare le conclusioni con esempi specifici
- Considerare il contesto temporale completo
- Suggerire possibili sviluppi futuri"""

    def _extract_thread_stats(self, context: str) -> dict:
        """Estrae statistiche dettagliate dal contesto per supportare l'analisi."""
        import re
        from collections import defaultdict
        from datetime import datetime

        stats = {
            "threads": defaultdict(dict),
            "users": defaultdict(list),
            "citations": defaultdict(list),
            "temporal_data": defaultdict(list),
            "keywords": defaultdict(int),
            "sentiment_trends": defaultdict(list)
        }

        # Pattern per estrarre informazioni
        thread_pattern = r"THREAD: (.*?)\nPosts trovati: (\d+)\nPosts dichiarati: (\d+)"
        post_pattern = r"\[(.*?)\] (.*?)\nSentiment: ([-\d.]+)\nKeywords: (.*?)\nContent:(.*?)(?=\n\[|\Z)"
        citation_pattern = r"(.*?) said:(.*?)Click to expand\.\.\.(.*)"

        # Estrai informazioni sui thread
        thread_matches = re.finditer(thread_pattern, context, re.DOTALL)
        for match in thread_matches:
            title, found, declared = match.groups()
            thread_id = title.strip()
            stats["threads"][thread_id] = {
                "posts_found": int(found),
                "posts_declared": int(declared),
                "discrepancy": int(declared) - int(found)
            }

        # Estrai informazioni sui post
        post_matches = re.finditer(post_pattern, context, re.DOTALL)
        for match in post_matches:
            timestamp, author, sentiment, keywords, content = match.groups()
            
            # Analisi temporale
            try:
                dt = datetime.strptime(timestamp.strip(), "%Y-%m-%dT%H:%M:%S%z")
                stats["temporal_data"][dt.strftime("%Y-%m")].append({
                    "author": author.strip(),
                    "sentiment": float(sentiment)
                })
            except ValueError:
                pass

            # Analisi utenti
            stats["users"][author.strip()].append({
                "timestamp": timestamp.strip(),
                "sentiment": float(sentiment),
                "content_length": len(content.strip())
            })

            # Analisi keywords
            for kw in keywords.split(", "):
                if kw.strip():
                    stats["keywords"][kw.strip()] += 1

            # Analisi citazioni
            citation_match = re.search(citation_pattern, content)
            if citation_match:
                quoted_author, quoted_content, actual_content = citation_match.groups()
                stats["citations"].append({
                    "quoting_author": author.strip(),
                    "quoted_author": quoted_author.strip(),
                    "timestamp": timestamp.strip(),
                    "sentiment": float(sentiment)
                })

            # Analisi sentiment
            stats["sentiment_trends"][timestamp[:7]].append(float(sentiment))

        # Calcola statistiche aggregate
        stats["aggregated"] = {
            "total_threads": len(stats["threads"]),
            "total_users": len(stats["users"]),
            "total_citations": len(stats["citations"]),
            "top_keywords": sorted(stats["keywords"].items(), key=lambda x: x[1], reverse=True)[:10],
            "user_engagement": sorted(
                [(user, len(posts)) for user, posts in stats["users"].items()],
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "temporal_range": {
                "start": min(stats["temporal_data"].keys()) if stats["temporal_data"] else None,
                "end": max(stats["temporal_data"].keys()) if stats["temporal_data"] else None
            }
        }

        # Calcola trend del sentiment
        if stats["sentiment_trends"]:
            sorted_trends = sorted(stats["sentiment_trends"].items())
            stats["aggregated"]["sentiment_evolution"] = [
                {
                    "period": period,
                    "avg_sentiment": sum(sentiments)/len(sentiments) if sentiments else 0
                }
                for period, sentiments in sorted_trends
            ]

        return dict(stats)  # Converti defaultdict in dict normale

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