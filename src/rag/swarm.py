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
            model_name="gpt-3.5-turbo-16k",
            temperature=0.3,
            api_key=st.secrets["OPENAI_API_KEY"]
        )
        
        # Modello per sintesi finale
        self.synthesis_llm = ChatOpenAI(
            model_name="gpt-4-turbo-preview",
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
    
    async def analyze_batch(self, batch: List[Document]) -> str:
        """Analizza un singolo batch di documenti."""
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
            return response.content
            
        except Exception as e:
            logger.error(f"Error analyzing batch: {str(e)}")
            return f"Error in batch analysis: {str(e)}"
    
    async def process_documents(self, documents: List[Document]) -> str:
        """Processa tutti i documenti usando lo swarm di analisi."""
        try:
            # Crea i batch
            batches = self.create_batches(documents)
            logger.info(f"Created {len(batches)} batches for analysis")
            
            # Analizza i batch in parallelo
            analysis_tasks = [self.analyze_batch(batch) for batch in batches]
            batch_analyses = await asyncio.gather(*analysis_tasks)
            
            # Sintetizza i risultati
            synthesis_prompt = "\n\n".join([
                "=== Analisi Batch ===\n" + analysis
                for analysis in batch_analyses
            ])
            
            messages = [
                SystemMessage(content="""Sintetizza le analisi dei vari batch in un'unica analisi coerente.
                Identifica:
                - Temi ricorrenti
                - Evoluzione temporale delle discussioni
                - Collegamenti tra thread diversi
                - Conclusioni generali
                Fornisci una panoramica completa e dettagliata."""),
                HumanMessage(content=f"Analisi da sintetizzare:\n{synthesis_prompt}")
            ]
            
            final_response = await self.synthesis_llm.ainvoke(messages)
            return final_response.content
            
        except Exception as e:
            logger.error(f"Error in swarm processing: {str(e)}")
            return f"Error in analysis: {str(e)}"