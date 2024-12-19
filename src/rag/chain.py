from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from typing import Dict, Any, List
import streamlit as st
import logging

logger = logging.getLogger(__name__)

class RAGChain:
    def __init__(self, retriever, model_name: str = "gpt-3.5-turbo", temperature: float = 0):
        self.retriever = retriever
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            api_key=st.secrets["OPENAI_API_KEY"]
        )
        self.setup_prompts()
        
    def setup_prompts(self):
        """Configura i template delle prompt per diversi scenari."""
        self.base_prompt = """Sei un assistente esperto nell'analisi di discussioni di forum. 
Segui questi principi nel fornire le tue risposte:

1. ACCURATEZZA:
- Rispondi SOLO basandoti sui documenti forniti
- Non fare supposizioni o aggiunte
- Se le informazioni non sono sufficienti, dichiaralo esplicitamente
- Cita specifiche parti del testo quando possibile

2. COMPRENSIONE DEL CONTESTO:
- Considera la discussione nel suo insieme
- Mantieni la cronologia e il flusso della conversazione
- Identifica e collega temi ricorrenti
- Evidenzia le relazioni tra i diversi post

3. STRUTTURA DELLA RISPOSTA:
- Inizia con una chiara panoramica
- Organizza le informazioni in modo logico
- Usa punti elenco solo quando appropriato
- Fornisci dettagli specifici con citazioni quando rilevante

4. FOCUS SULLA QUERY:
- Indirizza direttamente la domanda posta
- Mantieni la risposta pertinente
- Evita divagazioni non necessarie

Documenti di contesto:
{context}

Domanda: {query}

Fornisci una risposta strutturata, precisa e basata esclusivamente sui documenti forniti."""

        self.summary_prompt = """Analizza questa discussione del forum e fornisci un riassunto strutturato.
Focus su:
- Argomento principale e scopo della discussione
- Punti chiave sollevati
- Eventuali conclusioni o consensi raggiunti
- Temi ricorrenti o pattern nella discussione

Documenti:
{context}

Produci un riassunto coeso e informativo."""

        self.stat_prompt = """Analizza questi dati statistici del forum e fornisci un'interpretazione chiara.
Evidenzia:
- Numeri chiave
- Tendenze significative
- Confronti rilevanti
- Insight notevoli

Dati:
{context}

Fornisci un'analisi concisa e significativa."""

    def select_prompt(self, query: str) -> str:
        """Seleziona il template appropriato in base al tipo di query."""
        if any(keyword in query.lower() for keyword in ['riassunto', 'riassumi', 'sintetizza']):
            return self.summary_prompt
        elif any(keyword in query.lower() for keyword in ['statistiche', 'numeri', 'quanti']):
            return self.stat_prompt
        return self.base_prompt

    def process_documents(self, docs: List[Document]) -> str:
        """Processa e formatta i documenti per il contesto."""
        if not docs:
            return "Nessun documento disponibile."
            
        context = []
        for doc in docs:
            if doc.metadata.get("type") == "error":
                return doc.page_content
                
            content = f"""
---
{doc.page_content}
Similarità: {doc.metadata.get('similarity_score', 'N/A')}
---
"""
            context.append(content)
            
        return "\n".join(context)

    def __call__(self, query_input: Dict[str, str]) -> Dict[str, str]:
        """Esegue la catena RAG completa."""
        try:
            query = query_input.get("query", "") if isinstance(query_input, dict) else query_input
            
            # Recupera documenti rilevanti
            docs = self.retriever.get_relevant_documents(query)
            if not docs:
                return {
                    "result": "Mi dispiace, non ho trovato informazioni sufficienti per rispondere alla tua domanda.",
                    "documents": []
                }
            
            # Processa il contesto
            context = self.process_documents(docs)
            
            # Seleziona e formatta la prompt
            prompt_template = self.select_prompt(query)
            formatted_prompt = prompt_template.format(context=context, query=query)
            
            # Genera la risposta
            messages = [
                SystemMessage(content="Sei un assistente esperto nell'analisi di discussioni di forum."),
                HumanMessage(content=formatted_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            return {
                "result": response.content if hasattr(response, 'content') else str(response),
                "documents": docs
            }
            
        except Exception as e:
            logger.error(f"Error in RAG chain: {str(e)}", exc_info=True)
            return {
                "result": f"Mi dispiace, c'è stato un errore nell'elaborazione della risposta: {str(e)}",
                "documents": []
            }

def setup_rag_chain(retriever):
    """Funzione helper per creare un'istanza della catena RAG."""
    return RAGChain(retriever)