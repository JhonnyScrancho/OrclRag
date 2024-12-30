import asyncio
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import streamlit as st
import logging
from datetime import datetime
import tiktoken
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .templates import (
    template, 
    analyzer_role_desc, 
    analyzer_context_section,
    analyzer_instructions,
    synthesizer_role_desc,
    synthesizer_context_section,
    synthesizer_instructions
)

logger = logging.getLogger(__name__)

class OpenAISwarm:
    def __init__(self):
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
            if not api_key:
                raise ValueError("OpenAI API key not found")

            # Analyzer agents use 16k model for more context
            self.analyzer_llm = ChatOpenAI(
                model_name="gpt-3.5-turbo-16k",
                temperature=0.3,
                api_key=api_key,
                max_tokens=4000,
                request_timeout=60
            )
            
            # Synthesizer agent uses standard model
            self.synthesizer_llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.3,
                api_key=api_key,
                max_tokens=3000,
                request_timeout=30
            )
            
            # Token management
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
            self.MAX_TOKENS_PER_REQUEST = 14000  # Safe limit for gpt-3.5-turbo-16k
            self.MAX_SYNTHESIS_TOKENS = 10000
            self.MAX_RETRIES = 3
            self.MAX_PARALLEL_REQUESTS = 5
            
        except Exception as e:
            logger.error(f"Error initializing OpenAISwarm: {str(e)}")
            raise

    def count_tokens(self, text: str) -> int:
        """Conta i token in un testo usando tiktoken."""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {str(e)}")
            return len(text) // 4  # Fallback approssimativo

    def truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        """Tronca il testo per rispettare il limite di token mantenendo frasi complete."""
        try:
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= max_tokens:
                return text

            # Decodifica i token fino al limite e trova l'ultimo punto
            truncated = self.tokenizer.decode(tokens[:max_tokens])
            last_period = truncated.rfind('.')
            
            if last_period > max_tokens * 0.7:  # Se troviamo un punto dopo il 70% del testo
                return truncated[:last_period + 1]
            return truncated
            
        except Exception as e:
            logger.error(f"Error truncating text: {str(e)}")
            return text[:max_tokens * 4]  # Fallback approssimativo

    def split_documents_for_agents(self, documents: List[Document], num_agents: int) -> List[List[Document]]:
        """Divide i documenti equamente tra gli agenti."""
        try:
            if not documents:
                return []

            # Ordina i documenti per timestamp per mantenere la coerenza temporale
            sorted_docs = sorted(
                documents,
                key=lambda x: datetime.fromisoformat(x.metadata.get('post_time', '1970-01-01T00:00:00+00:00'))
            )

            # Calcola la dimensione di ogni porzione
            chunk_size = len(sorted_docs) // num_agents
            if chunk_size == 0:
                chunk_size = 1

            # Divide i documenti in porzioni
            agent_docs = []
            for i in range(0, len(sorted_docs), chunk_size):
                end_idx = i + chunk_size
                if i >= len(sorted_docs):
                    break
                if end_idx >= len(sorted_docs) or len(agent_docs) == num_agents - 1:
                    agent_docs.append(sorted_docs[i:])
                    break
                agent_docs.append(sorted_docs[i:end_idx])

            return agent_docs

        except Exception as e:
            logger.error(f"Error splitting documents for agents: {str(e)}")
            raise

    def get_thread_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """Calcola le statistiche dei thread."""
        try:
            thread_ids = set()
            thread_titles = set()
            posts_per_thread = {}
            
            for doc in documents:
                thread_id = doc.metadata.get('thread_id', 'unknown')
                thread_title = doc.metadata.get('thread_title', 'Unknown Thread')
                
                thread_ids.add(thread_id)
                thread_titles.add(thread_title)
                
                if thread_id not in posts_per_thread:
                    posts_per_thread[thread_id] = 0
                posts_per_thread[thread_id] += 1
            
            return {
                "num_threads": len(thread_ids),
                "thread_titles": list(thread_titles),
                "total_posts": len(documents),
                "avg_posts_per_thread": len(documents) / len(thread_ids) if thread_ids else 0
            }
        except Exception as e:
            logger.error(f"Error calculating thread stats: {str(e)}")
            return {
                "num_threads": 0,
                "thread_titles": [],
                "total_posts": 0,
                "avg_posts_per_thread": 0
            }

    def format_documents(self, documents: List[Document]) -> str:
        """Formatta i documenti per l'analisi."""
        try:
            # Calcola le statistiche prima di formattare
            stats = self.get_thread_stats(documents)
            
            # Aggiungi l'header con le statistiche
            header = f"""📊 Statistiche Globali:
- Numero totale di thread: {stats['num_threads']}
- Numero totale di post: {stats['total_posts']}
- Media post per thread: {stats['avg_posts_per_thread']:.1f}

🧵 Thread analizzati:
{chr(10).join(f"- {title}" for title in stats['thread_titles'])}

---
📝 Dettaglio dei post:
"""
            
            formatted_posts = []
            total_tokens = 0
            
            for doc in documents:
                thread_title = doc.metadata.get('thread_title', 'Unknown Thread')
                author = doc.metadata.get('author', 'Unknown')
                time = doc.metadata.get('post_time', '')
                content = doc.page_content.strip()
                keywords = doc.metadata.get('keywords', [])
                sentiment = doc.metadata.get('sentiment', 0.0)
                
                post = f"""Thread: {thread_title}
Author: {author}
Time: {time}
Keywords: {', '.join(keywords) if keywords else 'N/A'}
Sentiment: {sentiment}
Content: {content}
---"""
                
                post_tokens = self.count_tokens(post)
                
                # Se il post non supera il limite, aggiungilo
                if total_tokens + post_tokens <= self.MAX_TOKENS_PER_REQUEST * 0.7:  # 70% del limite per sicurezza
                    formatted_posts.append(post)
                    total_tokens += post_tokens
                else:
                    break
            
            return header + "\n\n".join(formatted_posts)
            
        except Exception as e:
            logger.error(f"Error formatting documents: {str(e)}")
            raise

    async def analyze_with_agent(self, 
                           documents: List[Document], 
                           agent_id: int,
                           query: str,
                           retry_count: int = 0) -> Optional[str]:
        """Analizza documenti con un singolo agente."""
        if retry_count >= self.MAX_RETRIES:
            logger.error(f"Max retries ({self.MAX_RETRIES}) reached for agent #{agent_id}")
            return None

        try:
            formatted_content = self.format_documents(documents)
            if not formatted_content.strip():
                logger.warning(f"Agent #{agent_id}: Empty formatted content")
                return None

            # Calcola i token del contenuto formattato
            content_tokens = self.count_tokens(formatted_content)
            logger.info(f"Agent #{agent_id}: Content tokens: {content_tokens}")

            # Prepara il messaggio di sistema
            system_message = template.format(
                agent_id=agent_id + 1,
                role_desc=analyzer_role_desc,
                context_section=analyzer_context_section.format(context="[CONTEXT]"),  # Placeholder
                query=query,
                role_instructions=analyzer_instructions
            )
            system_tokens = self.count_tokens(system_message)
            logger.info(f"Agent #{agent_id}: System message tokens: {system_tokens}")

            # Verifica se il totale dei token è entro il limite
            total_tokens = content_tokens + system_tokens
            available_tokens = self.MAX_TOKENS_PER_REQUEST - 4000  # Riserva 4000 token per la risposta
            
            if total_tokens > available_tokens:
                logger.warning(f"Agent #{agent_id}: Token limit exceeded. Total: {total_tokens}, Available: {available_tokens}")
                # Tronca il contenuto se necessario
                max_content_tokens = available_tokens - system_tokens
                formatted_content = self.truncate_to_token_limit(formatted_content, max_content_tokens)
                logger.info(f"Agent #{agent_id}: Content truncated to {self.count_tokens(formatted_content)} tokens")

            # Costruisci il messaggio finale
            system_message = template.format(
                agent_id=agent_id + 1,
                role_desc=analyzer_role_desc,
                context_section=analyzer_context_section.format(context=formatted_content),
                query=query,
                role_instructions=analyzer_instructions
            )

            messages = [
                SystemMessage(content=system_message)
            ]
            
            logger.info(f"Agent #{agent_id}: Sending request to OpenAI")
            response = await self.analyzer_llm.ainvoke(messages)
            
            if not response or not response.content:
                logger.warning(f"Agent #{agent_id}: Empty response from OpenAI")
                return None
                
            logger.info(f"Agent #{agent_id}: Successfully received response")
            return response.content
            
        except Exception as e:
            logger.error(f"Error in agent #{agent_id} analysis (attempt {retry_count + 1}): {str(e)}")
            if retry_count < self.MAX_RETRIES:
                await asyncio.sleep(2 ** retry_count)
                return await self.analyze_with_agent(documents, agent_id, query, retry_count + 1)
            return None

    async def synthesize_analyses(self, 
                            analyses: List[str], 
                            query: str,
                            retry_count: int = 0) -> str:
        """Sintetizza le analisi degli agenti."""
        if retry_count >= self.MAX_RETRIES:
            return "Non è stato possibile completare la sintesi dei risultati."

        try:
            valid_analyses = [a for a in analyses if a and a.strip()]
            if not valid_analyses:
                return "Nessuna analisi valida da sintetizzare."

            synthesis_text = "\n\n".join([
                f"--- Analisi Agente #{i+1} ---\n{analysis}"
                for i, analysis in enumerate(valid_analyses)
            ])

            messages = [
                SystemMessage(content=template.format(
                    agent_id="S",
                    role_desc=synthesizer_role_desc,
                    context_section=synthesizer_context_section.format(context=synthesis_text),
                    query=query,
                    role_instructions=synthesizer_instructions
                )),
                HumanMessage(content=synthesis_text)
            ]
            
            response = await self.synthesizer_llm.ainvoke(messages)
            if not response or not response.content:
                raise ValueError("Empty response from synthesis")
                
            return response.content
            
        except Exception as e:
            logger.error(f"Error in synthesis (attempt {retry_count + 1}): {str(e)}")
            if retry_count < self.MAX_RETRIES:
                await asyncio.sleep(2 ** retry_count)
                return await self.synthesize_analyses(analyses, query, retry_count + 1)
            return "Errore nella sintesi dei risultati."

    async def process_documents(self, 
                              documents: List[Document], 
                              query: str,
                              status_container) -> str:
        """Processa i documenti usando il sistema multi-agente."""
        try:
            if not documents:
                return "Nessun documento da analizzare."

            # Get number of agents from session state
            num_agents = st.session_state.get('num_agents', 3)
                
            # Split documents between agents
            agent_docs = self.split_documents_for_agents(documents, num_agents)
            status_container.write(f"📦 Documenti divisi tra {len(agent_docs)} agenti")
            logger.info(f"Documents split between {len(agent_docs)} agents")
            
            # Create progress tracking
            progress_text = "🔄 Analisi in corso..."
            progress_bar = status_container.progress(0, text=progress_text)
            
            # Process documents with agents in parallel
            semaphore = asyncio.Semaphore(self.MAX_PARALLEL_REQUESTS)
            async def process_with_agent(docs, agent_id):
                async with semaphore:
                    result = await self.analyze_with_agent(docs, agent_id, query)
                    progress = (agent_id + 1) / len(agent_docs)
                    progress_bar.progress(progress, text=f"{progress_text} ({agent_id + 1}/{len(agent_docs)})")
                    
                    if result:
                        msg = f"✅ Agente #{agent_id + 1}: Analisi completata"
                        if st.session_state.show_agent_details:
                            msg += f"\n{result}\n---"
                        status_container.write(msg)
                    else:
                        status_container.warning(f"⚠️ Agente #{agent_id + 1}: Analisi completata con warning")
                    return result

            # Execute analyses in parallel
            tasks = [process_with_agent(docs, i) for i, docs in enumerate(agent_docs)]
            agent_results = await asyncio.gather(*tasks)

            # Filter valid results
            valid_results = [r for r in agent_results if r]
            if not valid_results:
                return "Nessun agente ha prodotto un'analisi valida."

            # Synthesize results
            status_container.write("🤖 Agente sintetizzatore al lavoro...")
            final_result = await self.synthesize_analyses(valid_results, query)

            status_container.write("🏁 Analisi completata!")
            return final_result

        except Exception as e:
            error_msg = f"Error in multi-agent processing: {str(e)}"
            logger.error(error_msg)
            status_container.error(f"❌ {error_msg}")
            return f"Errore nell'elaborazione: {str(e)}"