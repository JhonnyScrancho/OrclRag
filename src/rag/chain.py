from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import streamlit as st
import logging
from datetime import datetime
from .swarm import OpenAISwarm
from .templates import template
import asyncio

logger = logging.getLogger(__name__)

def setup_rag_chain(retriever, num_agents=3):
    """
    Configura la chain RAG.
    Args:
        retriever: Il retriever da utilizzare
        num_agents (int): Numero di agenti da utilizzare (default: 3)
    """
    try:
        def get_response(query):
            # Recupera i documenti rilevanti
            docs = retriever.get_relevant_documents(query)
            logger.info(f"📚 Recuperati {len(docs)} documenti dal database")
            
            # Processa i documenti con lo swarm
            swarm = OpenAISwarm(num_agents=num_agents)
            response = swarm.process_documents(docs)
            
            return response
            
        return {"get_response": get_response}
        
    except Exception as e:
        logger.error(f"Error setting up RAG chain: {str(e)}")
        raise