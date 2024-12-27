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
        # Modello per analisi dei batch
        self.analysis_llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.3,
            api_key=st.secrets["OPENAI_API_KEY"]
        )
        
        # Modello per sintesi finale
        self.synthesis_llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.3,
            api_key=st.secrets["OPENAI_API_KEY"]
        )
        
        # Dimensione target per ogni batch (in caratteri)
        self.BATCH_TARGET_SIZE = 40000  # ~10k tokens
        
    def estimate_tokens(self, text: str) -> int:
        """Stima approssimativa dei token basata sui caratteri."""
        return len(text) // 4
    
    def create_batches(self, documents: List[Document]) -> List[List[Document]]:
        """Divide i documenti in batch gestibili mantenendo i thread uniti."""
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
                batches.append(current_batch)
                current_batch = []
                current_size = 0
            
            current_batch.append(doc)
            current_size += doc_size
            current_thread = thread_id
        
        if current_batch:
            batches.append(current_batch)
            
        return batches
    
    def format_batch(self, batch: List[Document]) -> str:
        """Formatta un batch di documenti per l'analisi."""
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
            
            for post in thread_data['posts']:
                formatted_text += f"[{post['time']}] {post['author']}:\n{post['content']}\n---\n"
        
        return formatted_text
    
    async def process_documents(self, documents: List[Document], status_container) -> str:
        """Processa tutti i documenti usando lo swarm."""
        try:
            # Crea i batch
            batches = self.create_batches(documents)
            status_container.write(f"📦 Creati {len(batches)} batch per l'analisi")
            logger.info(f"Created {len(batches)} batches")
            
            # Crea la progress bar
            progress_text = "🔄 Analisi batch in corso..."
            progress_bar = status_container.progress(0, text=progress_text)
            
            # Analizza tutti i batch in parallelo
            tasks = [self.analyze_batch(batch) for batch in batches]
            batch_results = []
            
            for i, task in enumerate(asyncio.as_completed(tasks)):
                result = await task
                batch_results.append(result)
                # Aggiorna progress bar
                progress = (i + 1) / len(tasks)
                progress_bar.progress(progress, text=f"{progress_text} ({i + 1}/{len(tasks)})")
                status_container.write(f"✅ Completato batch {i + 1}/{len(tasks)}")
            
            # Sintetizza i risultati
            if len(batch_results) == 1:
                status_container.write("🏁 Analisi completata!")
                return batch_results[0]
            
            status_container.write("🔄 Sintetizzando i risultati...")
            synthesis_prompt = "\n\n".join(batch_results)
            
            messages = [
                SystemMessage(content="""Sintetizza le analisi in un'unica risposta coerente.
                Mantieni:
                - Collegamenti tra thread diversi
                - Evoluzione temporale delle discussioni
                - Citazioni rilevanti"""),
                HumanMessage(content=synthesis_prompt)
            ]
            
            # Fixed: Using self.synthesis_llm instead of self.batch_llm
            final_response = await self.synthesis_llm.ainvoke(messages)
            status_container.write("🏁 Sintesi completata!")
            return final_response.content
            
        except Exception as e:
            logger.error(f"Error in swarm processing: {str(e)}")
            status_container.error(f"❌ Errore nell'elaborazione: {str(e)}")
            raise