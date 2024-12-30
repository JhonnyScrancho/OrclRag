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

logger = logging.getLogger(__name__)

class OpenAISwarm:
    def __init__(self):
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
            if not api_key:
                raise ValueError("OpenAI API key not found")

            # Inizializza i modelli con configurazioni ottimizzate
            self.analysis_llm = ChatOpenAI(
                model_name="gpt-3.5-turbo-16k",  # Usa il modello con contesto più ampio
                temperature=0.3,
                api_key=api_key,
                max_tokens=4000,
                request_timeout=60
            )
            
            self.synthesis_llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",  # Modello standard per la sintesi
                temperature=0.3,
                api_key=api_key,
                max_tokens=3000,
                request_timeout=30
            )
            
            # Inizializza l'encoder per il conteggio dei token
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
            
            # Configurazione ottimizzata per i batch
            self.MAX_BATCH_TOKENS = 12000  # ~75% del limite di 16k
            self.MAX_SYNTHESIS_TOKENS = 10000  # Per la sintesi finale
            self.MIN_BATCH_SIZE = 3  # Numero minimo di documenti per batch
            self.MAX_RETRIES = 3  # Tentativi massimi per ogni operazione
            self.MAX_PARALLEL_REQUESTS = 5  # Limite di richieste parallele
            
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

    def estimate_total_tokens(self, documents: List[Document]) -> int:
        """Stima il numero totale di token nei documenti."""
        return sum(self.count_tokens(doc.page_content) for doc in documents)

    def create_batches(self, documents: List[Document]) -> List[List[Document]]:
        """Crea batch di documenti ottimizzati per il processing parallelo."""
        try:
            # Ordina i documenti per thread e timestamp
            sorted_docs = sorted(
                documents,
                key=lambda x: (
                    x.metadata.get('thread_id', ''),
                    datetime.fromisoformat(x.metadata.get('post_time', '1970-01-01T00:00:00+00:00'))
                )
            )
            
            batches = []
            current_batch = []
            current_tokens = 0
            current_thread = None
            
            for doc in sorted_docs:
                thread_id = doc.metadata.get('thread_id')
                doc_tokens = self.count_tokens(doc.page_content)
                
                # Condizioni per creare un nuovo batch
                new_batch_needed = (
                    (current_thread and thread_id != current_thread) or  # Cambio thread
                    (current_tokens + doc_tokens > self.MAX_BATCH_TOKENS) or  # Limite token
                    (len(current_batch) >= self.MIN_BATCH_SIZE)  # Dimensione minima
                )
                
                if new_batch_needed and current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                
                current_batch.append(doc)
                current_tokens += doc_tokens
                current_thread = thread_id
            
            # Aggiungi l'ultimo batch se non vuoto
            if current_batch:
                batches.append(current_batch)
            
            # Bilancia i batch se necessario
            if len(batches) > 1:
                balanced_batches = self._balance_batches(batches)
                return balanced_batches
            
            return batches
            
        except Exception as e:
            logger.error(f"Error creating batches: {str(e)}")
            raise

    def _balance_batches(self, batches: List[List[Document]]) -> List[List[Document]]:
        """Bilancia i batch per una dimensione più uniforme."""
        try:
            # Calcola la dimensione media dei batch
            batch_sizes = [sum(self.count_tokens(doc.page_content) for doc in batch) 
                         for batch in batches]
            avg_size = np.mean(batch_sizes)
            
            # Identifica batch troppo grandi o troppo piccoli
            balanced = []
            for batch in batches:
                batch_tokens = sum(self.count_tokens(doc.page_content) for doc in batch)
                
                if batch_tokens > avg_size * 1.5:  # Batch troppo grande
                    # Dividi il batch
                    mid = len(batch) // 2
                    balanced.extend([batch[:mid], batch[mid:]])
                elif batch_tokens < avg_size * 0.5:  # Batch troppo piccolo
                    # Accumula per il prossimo batch
                    if balanced and len(balanced[-1]) < self.MIN_BATCH_SIZE:
                        balanced[-1].extend(batch)
                    else:
                        balanced.append(batch)
                else:
                    balanced.append(batch)
            
            return balanced
            
        except Exception as e:
            logger.error(f"Error balancing batches: {str(e)}")
            return batches

    def format_batch(self, batch: List[Document]) -> str:
        """Formatta un batch di documenti per l'analisi."""
        try:
            formatted_posts = []
            total_tokens = 0
            
            for doc in batch:
                # Estrai e valida i metadati
                thread_title = doc.metadata.get('thread_title', 'Unknown Thread')
                author = doc.metadata.get('author', 'Unknown')
                time = doc.metadata.get('post_time', '')
                content = doc.page_content.strip()
                
                # Formatta il post
                post = f"""Thread: {thread_title}
Author: {author}
Time: {time}
Content: {content}
---"""
                
                post_tokens = self.count_tokens(post)
                
                # Se il post non supera il limite, aggiungilo
                if total_tokens + post_tokens <= self.MAX_BATCH_TOKENS:
                    formatted_posts.append(post)
                    total_tokens += post_tokens
                else:
                    break
            
            return "\n\n".join(formatted_posts)
            
        except Exception as e:
            logger.error(f"Error formatting batch: {str(e)}")
            raise

    async def analyze_batch(self, batch: List[Document], retry_count: int = 0) -> Optional[str]:
        """Analizza un batch con retry automatico."""
        if retry_count >= self.MAX_RETRIES:
            logger.error(f"Max retries ({self.MAX_RETRIES}) reached for batch analysis")
            return None

        try:
            formatted_content = self.format_batch(batch)
            if not formatted_content.strip():
                return None

            messages = [
                SystemMessage(content="""Analizza questi post del forum. Fornisci un'analisi concisa che includa:
                1. Punti chiave della discussione
                2. Opinioni principali espresse
                3. Collegamenti tra i post
                Mantieni l'analisi breve e rilevante."""),
                HumanMessage(content=formatted_content)
            ]
            
            response = await self.analysis_llm.ainvoke(messages)
            return response.content if response else None
            
        except Exception as e:
            logger.error(f"Error in batch analysis (attempt {retry_count + 1}): {str(e)}")
            # Retry con backoff esponenziale
            if retry_count < self.MAX_RETRIES:
                await asyncio.sleep(2 ** retry_count)
                return await self.analyze_batch(batch, retry_count + 1)
            return None

    async def synthesize_results(self, results: List[str], retry_count: int = 0) -> str:
        """Sintetizza i risultati delle analisi con gestione degli errori."""
        if retry_count >= self.MAX_RETRIES:
            return "Non è stato possibile completare la sintesi dei risultati."

        try:
            # Filtra e limita i risultati
            valid_results = [r for r in results if r and r.strip()]
            if not valid_results:
                return "Nessun risultato valido da sintetizzare."

            # Prepara il contenuto per la sintesi
            synthesis_text = ""
            current_tokens = 0
            
            for result in valid_results:
                result_tokens = self.count_tokens(result)
                if current_tokens + result_tokens <= self.MAX_SYNTHESIS_TOKENS:
                    synthesis_text += f"\n\n---\n\n{result}"
                    current_tokens += result_tokens
                else:
                    break

            messages = [
                SystemMessage(content="""Sintetizza le analisi in una risposta unica e coerente.
                Mantieni solo le informazioni più rilevanti e significative.
                La risposta deve essere concisa e ben strutturata."""),
                HumanMessage(content=synthesis_text)
            ]
            
            response = await self.synthesis_llm.ainvoke(messages)
            if not response or not response.content:
                raise ValueError("Empty response from synthesis")
                
            return response.content
            
        except Exception as e:
            logger.error(f"Error in synthesis (attempt {retry_count + 1}): {str(e)}")
            if retry_count < self.MAX_RETRIES:
                await asyncio.sleep(2 ** retry_count)
                return await self.synthesize_results(results, retry_count + 1)
            return "Errore nella sintesi dei risultati."

    async def process_documents(self, documents: List[Document], status_container) -> str:
        """Processa i documenti utilizzando lo swarm con gestione parallela ottimizzata."""
        try:
            if not documents:
                return "Nessun documento da analizzare."

            # Crea i batch
            batches = self.create_batches(documents)
            if not batches:
                return "Impossibile creare batch dai documenti."

            status_container.write(f"📦 Creati {len(batches)} batch per l'analisi")
            logger.info(f"Created {len(batches)} batches for processing")

            # Prepara la progress bar
            progress_text = "🔄 Analisi batch in corso..."
            progress_bar = status_container.progress(0, text=progress_text)

            # Processa i batch in parallelo con limite di concorrenza
            semaphore = asyncio.Semaphore(self.MAX_PARALLEL_REQUESTS)
            async def process_with_semaphore(batch, index):
                async with semaphore:
                    result = await self.analyze_batch(batch)
                    progress = (index + 1) / len(batches)
                    progress_bar.progress(progress, text=f"{progress_text} ({index + 1}/{len(batches)})")
                    if result:
                        status_container.write(f"✅ Completato batch {index + 1}/{len(batches)}")
                    else:
                        status_container.warning(f"⚠️ Batch {index + 1}/{len(batches)} completato con warning")
                    return result

            # Esegui le analisi in parallelo
            tasks = [process_with_semaphore(batch, i) for i, batch in enumerate(batches)]
            batch_results = await asyncio.gather(*tasks)

            # Filtra i risultati validi
            valid_results = [r for r in batch_results if r]
            if not valid_results:
                return "Non è stato possibile analizzare i documenti."

            # Sintetizza i risultati
            status_container.write("🔄 Sintetizzando i risultati...")
            final_result = await self.synthesize_results(valid_results)

            status_container.write("🏁 Analisi completata!")
            return final_result

        except Exception as e:
            error_msg = f"Error in swarm processing: {str(e)}"
            logger.error(error_msg)
            status_container.error(f"❌ {error_msg}")
            return f"Errore nell'elaborazione: {str(e)}"