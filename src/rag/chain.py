from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
import streamlit as st
import json
import logging

logger = logging.getLogger(__name__)

def setup_rag_chain(retriever):
    """Configura e restituisce una chain RAG avanzata."""
    llm = ChatOpenAI(
        model_name="gpt-4-turbo-preview",
        temperature=0.7,
        api_key=st.secrets["OPENAI_API_KEY"]
    )
    
    template = """Sei un analista esperto di forum online con accesso a dati dettagliati sulle discussioni.
Hai piena libertà di analizzare e interpretare i dati forniti nel modo più appropriato.

LINEE GUIDA PER L'ANALISI:

1. CONTENUTO E CONTESTO
- Analizza il significato e il contesto della discussione
- Identifica i temi principali e secondari
- Considera il tono e lo stile della conversazione

2. METRICHE E TREND
- Esamina i sentiment e la loro evoluzione
- Analizza le keywords e la loro rilevanza
- Valuta i pattern di interazione tra utenti

3. DINAMICHE SOCIALI
- Osserva i ruoli degli utenti nella discussione
- Identifica leader e partecipanti chiave
- Analizza la qualità e profondità delle interazioni

4. INSIGHT SPECIFICI
- Cerca pattern non ovvi
- Identifica connessioni interessanti
- Evidenzia aspetti unici o notevoli

I dati completi della discussione sono strutturati come segue:
{context}

Domanda dell'utente: {query}

ISTRUZIONI PER LA RISPOSTA:
1. Fornisci un'analisi approfondita e pertinente
2. Usa dati concreti per supportare le tue osservazioni
3. Evidenzia insight non ovvi quando rilevanti
4. Mantieni un tono professionale ma accessibile
5. Rispondi in italiano in modo naturale e discorsivo

Se la domanda è specifica (es. numero di post), fornisci prima la risposta diretta e poi eventuali insight aggiuntivi se rilevanti."""
    
    def get_response(query_input):
        """Processa la query e genera una risposta."""
        try:
            query = query_input.get("query", "") if isinstance(query_input, dict) else query_input
            docs = retriever.get_relevant_documents(query)
            
            if not docs:
                return {
                    "result": "Mi dispiace, non ho trovato dati sufficienti per rispondere alla tua domanda."
                }
            
            # Estrai l'analisi ricca dal metadata
            rich_analysis = docs[0].metadata.get("analysis", {})
            if not rich_analysis:
                return {
                    "result": "Non sono riuscito a recuperare un'analisi dettagliata dei dati."
                }
            
            # Formatta il contesto in modo naturale
            context = json.dumps(rich_analysis, indent=2, ensure_ascii=False)
            
            messages = [
                SystemMessage(content="""Sei un analista esperto di forum online. 
Il tuo obiettivo è fornire insight profondi e significativi, basati sui dati ma con un'interpretazione intelligente e naturale.
Usa la tua conoscenza per identificare pattern interessanti e connessioni non ovvie."""),
                HumanMessage(content=template.format(context=context, query=query))
            ]
            
            response = llm.invoke(messages)
            
            return {
                "result": response.content
            }
            
        except Exception as e:
            logger.error(f"Error in RAG chain: {str(e)}", exc_info=True)
            return {
                "result": f"Si è verificato un errore durante l'elaborazione della risposta: {str(e)}"
            }
    
    return get_response