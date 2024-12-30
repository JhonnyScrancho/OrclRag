import asyncio
from typing import List, Dict
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import streamlit as st
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class OpenAISwarm:
    def __init__(self):
        try:
            # Initialize LLMs with proper error handling
            api_key = st.secrets["OPENAI_API_KEY"]
            if not api_key:
                raise ValueError("OpenAI API key not found")

            self.analysis_llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.3,
                api_key=api_key
            )
            
            self.synthesis_llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.3,
                api_key=api_key
            )
            
            # Dimensione target per ogni batch (in caratteri)
            self.BATCH_TARGET_SIZE = 40000  # ~10k tokens
            
        except Exception as e:
            logger.error(f"Error initializing OpenAISwarm: {str(e)}")
            raise
    
    def estimate_tokens(self, text: str) -> int:
        """Stima approssimativa dei token basata sui caratteri."""
        return len(text) // 4
    
    def create_batches(self, documents: List[Document]) -> List[List[Document]]:
        """Divide i documenti in batch gestibili mantenendo i thread uniti."""
        try:
            # Ordina i documenti per thread_id e timestamp
            sorted_docs = sorted(
                documents,
                key=lambda x: (
                    x.metadata.get('thread_id', ''),
                    x.metadata.get('post_time', '')
                )
            )
            
            batches = []
            current_batch = []
            current_size = 0
            current_thread = None
            
            for doc in sorted_docs:
                thread_id = doc.metadata.get('thread_id')
                doc_size = len(doc.page_content)
                
                # Se cambia il thread o il batch è troppo grande
                if (current_thread and thread_id != current_thread and current_size > 0) or \
                   (current_size + doc_size > self.BATCH_TARGET_SIZE and current_thread != thread_id):
                    if current_batch:  # Verifica che il batch non sia vuoto
                        batches.append(current_batch)
                        current_batch = []
                        current_size = 0
                
                current_batch.append(doc)
                current_size += doc_size
                current_thread = thread_id
            
            # Aggiungi l'ultimo batch se non vuoto
            if current_batch:
                batches.append(current_batch)
                
            if not batches:
                raise ValueError("No valid batches created")
                
            return batches
            
        except Exception as e:
            logger.error(f"Error creating batches: {str(e)}")
            raise
    
    def format_batch(self, batch: List[Document]) -> str:
        """Formatta un batch di documenti per l'analisi."""
        try:
            if not batch:
                raise ValueError("Empty batch provided")
                
            threads = {}
            
            # Organizza i documenti per thread
            for doc in batch:
                thread_id = doc.metadata.get('thread_id', 'unknown')
                if thread_id not in threads:
                    threads[thread_id] = {
                        'title': doc.metadata.get('thread_title', 'Unknown Thread'),
                        'posts': []
                    }
                
                # Estrai le informazioni rilevanti dal post
                post_info = {
                    'author': doc.metadata.get('author', 'Unknown'),
                    'time': doc.metadata.get('post_time', ''),
                    'content': doc.page_content
                }
                threads[thread_id]['posts'].append(post_info)
            
            # Formatta il testo
            formatted_text = ""
            for thread_id, thread_data in threads.items():
                formatted_text += f"\n### Thread: {thread_data['title']}\n\n"
                
                # Ordina i post per timestamp se possibile
                try:
                    thread_data['posts'].sort(key=lambda x: datetime.fromisoformat(x['time']) if x['time'] else datetime.min)
                except (ValueError, TypeError):
                    logger.warning(f"Could not sort posts by timestamp in thread {thread_id}")
                
                for post in thread_data['posts']:
                    formatted_text += f"[{post['time']}] {post['author']}:\n{post['content']}\n---\n"
            
            if not formatted_text.strip():
                raise ValueError("No content generated from batch")
                
            return formatted_text
            
        except Exception as e:
            logger.error(f"Error formatting batch: {str(e)}")
            raise

    async def analyze_batch(self, batch: List[Document]) -> str:
        """Analizza un singolo batch di documenti."""
        if not self.analysis_llm:
            raise ValueError("LLM not properly initialized")

        try:
            formatted_content = self.format_batch(batch)
            
            messages = [
                SystemMessage(content="""Analizza questi thread del forum. 
                Identifica:
                - Argomenti principali discussi
                - Opinioni e esperienze riportate
                - Citazioni rilevanti
                - Collegamenti tra i post
                Fornisci un'analisi dettagliata ma concisa."""),
                HumanMessage(content=f"Thread da analizzare:\n{formatted_content}")
            ]
            
            response = await self.analysis_llm.ainvoke(messages)
            if not response or not hasattr(response, 'content'):
                raise ValueError("Invalid response from LLM")
                
            return response.content
            
        except Exception as e:
            logger.error(f"Error analyzing batch: {str(e)}")
            raise

    async def process_documents(self, documents: List[Document], status_container) -> str:
        """Processa tutti i documenti usando lo swarm."""
        if not documents:
            raise ValueError("No documents provided")

        try:
            # Crea i batch
            batches = self.create_batches(documents)
            if not batches:
                raise ValueError("No batches created from documents")
                
            status_container.write(f"📦 Creati {len(batches)} batch per l'analisi")
            logger.info(f"Created {len(batches)} batches")
            
            # Crea la progress bar
            progress_text = "🔄 Analisi batch in corso..."
            progress_bar = status_container.progress(0, text=progress_text)
            
            # Analizza tutti i batch in parallelo con gestione errori
            batch_results = []
            tasks = [self.analyze_batch(batch) for batch in batches]
            
            for i, task in enumerate(asyncio.as_completed(tasks)):
                try:
                    result = await task
                    if result:  # Verifica che il risultato non sia None
                        batch_results.append(result)
                    # Aggiorna progress bar
                    progress = (i + 1) / len(tasks)
                    progress_bar.progress(progress, text=f"{progress_text} ({i + 1}/{len(tasks)})")
                    status_container.write(f"✅ Completato batch {i + 1}/{len(tasks)}")
                except Exception as e:
                    logger.error(f"Error processing batch {i}: {str(e)}")
                    status_container.error(f"⚠️ Errore nel batch {i + 1}: {str(e)}")
            
            if not batch_results:
                raise ValueError("No valid results from batch processing")
            
            # Sintetizza i risultati
            if len(batch_results) == 1:
                status_container.write("🏁 Analisi completata!")
                return batch_results[0]
            
            status_container.write("🔄 Sintetizzando i risultati...")
            
            if not self.synthesis_llm:
                raise ValueError("Synthesis LLM not properly initialized")
                
            synthesis_prompt = "\n\n".join(batch_results)
            messages = [
                SystemMessage(content="""Sintetizza le analisi in un'unica risposta coerente.
                Mantieni:
                - Collegamenti tra thread diversi
                - Evoluzione temporale delle discussioni
                - Citazioni rilevanti"""),
                HumanMessage(content=synthesis_prompt)
            ]
            
            final_response = await self.synthesis_llm.ainvoke(messages)
            if not final_response or not hasattr(final_response, 'content'):
                raise ValueError("Invalid response from synthesis LLM")
                
            status_container.write("🏁 Sintesi completata!")
            return final_response.content
            
        except Exception as e:
            error_msg = f"Error in swarm processing: {str(e)}"
            logger.error(error_msg)
            status_container.error(f"❌ {error_msg}")
            raise