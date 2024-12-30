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
    def __init__(self, model="gpt-3.5-turbo-16k", num_agents=3):
        """
        Inizializza lo swarm di agenti.
        Args:
            model (str): Il modello da utilizzare
            num_agents (int): Numero di agenti da utilizzare (default: 3)
        """
        self.model = model
        self.num_agents = max(1, min(num_agents, 3))  # Limita tra 1 e 3 agenti
        self.llm = ChatOpenAI(
            model=model,
            temperature=0
        )
        
    def split_documents_for_agents(self, documents):
        """
        Divide i documenti tra gli agenti, raggruppando per thread_id.
        """
        # Raggruppa i documenti per thread_id
        thread_groups = {}
        for doc in documents:
            thread_id = doc.metadata.get('thread_id', '')
            if thread_id not in thread_groups:
                thread_groups[thread_id] = []
            thread_groups[thread_id].append(doc)
        
        # Se c'è un solo agente, restituisci tutti i documenti
        if self.num_agents == 1:
            return [list(documents)]
            
        # Altrimenti, distribuisci i thread tra gli agenti
        agent_docs = [[] for _ in range(self.num_agents)]
        for i, (_, thread_docs) in enumerate(thread_groups.items()):
            agent_idx = i % self.num_agents
            agent_docs[agent_idx].extend(thread_docs)
            
        return agent_docs

    def process_documents(self, documents):
        """
        Processa i documenti utilizzando lo swarm di agenti.
        """
        logger.info(f"📦 Documenti divisi tra {self.num_agents} agenti")
        
        # Dividi i documenti tra gli agenti
        agent_documents = self.split_documents_for_agents(documents)
        
        # Processa i documenti con ogni agente
        agent_results = []
        for i, docs in enumerate(agent_documents, 1):
            if not docs:  # Salta agenti senza documenti
                continue
                
            logger.info(f"🔄 Analisi in corso... ({i}/{self.num_agents})")
            
            try:
                formatted_docs = self.format_documents(docs)
                response = self.analyze_documents(formatted_docs, i)
                agent_results.append(response)
                logger.info(f"✅ Agente #{i}: Analisi completata")
            except Exception as e:
                logger.error(f"❌ Errore Agente #{i}: {str(e)}")
                agent_results.append(f"Errore nell'analisi: {str(e)}")
        
        # Sintetizza i risultati
        logger.info("🤖 Agente sintetizzatore al lavoro...")
        final_response = self.synthesize_results(agent_results)
        
        return final_response