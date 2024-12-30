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
from .templates import template

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
            self.base_template = template
            
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

    def format_documents(self, documents: List[Document]) -> str:
        """Formatta i documenti per l'analisi."""
        try:
            formatted_posts = []
            total_tokens = 0
            
            for doc in documents:
                thread_title = doc.metadata.get('thread_title', 'Unknown Thread')
                author = doc.metadata.get('author', 'Unknown')
                time = doc.metadata.get('post_time', '')
                content = doc.page_content.strip()
                
                post = f"""Thread: {thread_title}
Author: {author}
Time: {time}
Content: {content}
---"""
                
                post_tokens = self.count_tokens(post)
                
                # Se il post non supera il limite, aggiungilo
                if total_tokens + post_tokens <= self.MAX_TOKENS_PER_REQUEST * 0.7:  # 70% del limite per sicurezza
                    formatted_posts.append(post)
                    total_tokens += post_tokens
                else:
                    break
            
            return "\n\n".join(formatted_posts)
            
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
                return None

            messages = [
                SystemMessage(content=self.base_template.format(
                    agent_id=agent_id + 1,
                    query=query
                )),
                HumanMessage(content=formatted_content)
            ]
            
            response = await self.analyzer_llm.ainvoke(messages)
            return response.content if response else None
            
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
                SystemMessage(content=self.base_template.format(
                    role="sintetizzatore",
                    query=query
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