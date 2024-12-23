from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
import streamlit as st
import json
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

def setup_rag_chain(retriever):
    prompt_manager = PromptManager()
    """Configura una chain RAG semplificata che sfrutta le capacità di comprensione del LLM."""
    llm = ChatOpenAI(
        model_name="gpt-4-turbo-preview",
        temperature=0.3,
        api_key=st.secrets["OPENAI_API_KEY"]
    )
    
    template = """Sei un assistente esperto nell'analisi di conversazioni dei forum. Hai accesso ai dati di un thread del forum.
Nel rispondere, presta particolare attenzione a:
1. Identificare e utilizzare le citazioni presenti (formato "utente said: contenuto")
2. Comprendere il flusso della conversazione e chi risponde a chi
3. Interpretare correttamente il contesto temporale dei post
4. Evidenziare le citazioni rilevanti quando rispondi

Dati del forum:
{context}

Domanda: {query}

Fornisci una risposta concisa e pertinente in italiano, citando le parti rilevanti della conversazione quando appropriato.
Quando citi un post, usa il formato: "[Autore] ha scritto: '...'

REGOLE:
1. Rispondi SOLO a ciò che viene chiesto
2. Sii breve e diretto
3. Per domande numeriche, dai prima il numero e poi solo insight essenziali
4. Se rilevi citazioni, indicale esplicitamente
5. Non fare analisi non richieste """
    
    def get_response(query_input):
        try:
            # Gestisci sia input stringa che dizionario
            query = query_input.get("query", "") if isinstance(query_input, dict) else query_input
            
            # Ottieni i documenti rilevanti
            docs = retriever.get_all_documents()
            if not docs:
                return {"result": "Non ho trovato dati sufficienti per rispondere."}
            
            # Prepara il contesto come una sequenza temporale di post
            posts_context = []
            for doc in docs:
                post = {
                    "author": doc.metadata.get("author", "Unknown"),
                    "time": doc.metadata.get("post_time", "Unknown"),
                    "content": doc.metadata.get("text", ""),
                    "thread_title": doc.metadata.get("thread_title", "Unknown Thread")
                }
                posts_context.append(post)
            
            # Ordina i post per timestamp
            posts_context.sort(key=lambda x: x["time"])
            
            # Aggiungi il titolo del thread al contesto
            thread_title = posts_context[0]["thread_title"] if posts_context else "Unknown Thread"
            
            # Formatta il contesto come una conversazione
            context = f"Thread: {thread_title}\n\n" + "\n\n".join([
                f"[{post['time']}] {post['author']}:\n{post['content']}"
                for post in posts_context
            ])
            
            messages = [
                SystemMessage(content="Sei un assistente esperto nell'analisi di conversazioni dei forum."),
                HumanMessage(content=template.format(context=context, query=query))
            ]
            
            response = llm.invoke(messages)
            return {"result": response.content}
            
        except Exception as e:
            logger.error(f"Error in RAG chain: {str(e)}")
            return {"result": f"Errore nell'elaborazione: {str(e)}"}
    
    return get_response


class PromptManager:
    def __init__(self):
        self.function_definitions = {
            "extract_entities": {
                "name": "extract_entities",
                "description": "Estrae entità rilevanti dal testo",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "people": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Nomi di persone menzionate"
                        },
                        "topics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Argomenti principali discussi"
                        },
                        "sentiment": {
                            "type": "number",
                            "description": "Sentiment score da -1 a 1"
                        }
                    },
                    "required": ["people", "topics", "sentiment"]
                }
            },
            "analyze_conversation": {
                "name": "analyze_conversation",
                "description": "Analizza la struttura della conversazione",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "main_thread": {
                            "type": "string",
                            "description": "Tema principale della discussione"
                        },
                        "sub_threads": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Sotto-temi emersi"
                        },
                        "key_points": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Punti chiave discussi"
                        }
                    },
                    "required": ["main_thread", "sub_threads", "key_points"]
                }
            }
        }
        
        self.system_prompt = """Sei un assistente esperto nell'analisi di conversazioni dei forum, specializzato in:
1. Comprensione del contesto multilingua
2. Identificazione di relazioni tra post
3. Analisi della struttura conversazionale
4. Estrazione di informazioni chiave

CAPACITÀ:
- Identificare e collegare citazioni tra post
- Tracciare il flusso della conversazione
- Riconoscere topic e sotto-topic
- Analizzare il sentiment del discorso
- Estrarre entità rilevanti

REGOLE DI RISPOSTA:
1. Sii conciso ma completo
2. Usa evidenze dal testo
3. Mantieni la cronologia temporale
4. Cita direttamente quando rilevante
5. Identifica pattern conversazionali

GESTIONE MULTILINGUA:
- Riconosci e gestisci contenuti in più lingue
- Mantieni il contesto culturale
- Preserva sfumature linguistiche

FORMAT RISPOSTA:
1. Comprensione: sintesi della richiesta
2. Analisi: elaborazione strutturata
3. Evidenze: citazioni rilevanti
4. Conclusione: punti chiave
"""

    def build_prompt(self, context: str, query: str):
        """Costruisce il prompt completo"""
        return f"""CONTESTO:
{context}

QUERY:
{query}

Analizza il contenuto secondo le linee guida fornite.
"""

    def extract_entities_from_response(self, response: str):
        """Estrae entità dalla risposta usando function calling"""
        messages = [
            {"role": "system", "content": "Estrai entità dal seguente testo"},
            {"role": "user", "content": response}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            functions=[self.function_definitions["extract_entities"]],
            function_call={"name": "extract_entities"}
        )
        
        return json.loads(response.choices[0].function_call.arguments)

    def analyze_conversation_structure(self, thread_content: str):
        """Analizza la struttura della conversazione"""
        messages = [
            {"role": "system", "content": "Analizza la struttura della seguente conversazione"},
            {"role": "user", "content": thread_content}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            functions=[self.function_definitions["analyze_conversation"]],
            function_call={"name": "analyze_conversation"}
        )
        
        return json.loads(response.choices[0].function_call.arguments)

    def enhance_query_with_context(self, query: str, conversation_analysis: dict):
        """Arricchisce la query con il contesto della conversazione"""
        enhanced_query = f"""Query: {query}

Contesto Conversazione:
- Topic Principale: {conversation_analysis['main_thread']}
- Sotto-topic correlati: {', '.join(conversation_analysis['sub_threads'])}
- Punti chiave: {', '.join(conversation_analysis['key_points'])}

Per favore rispondi considerando questo contesto conversazionale."""
        
        return enhanced_query

    def format_response_with_evidence(self, response: str, citations: list):
        """Formatta la risposta includendo citazioni rilevanti"""
        formatted_response = response
        
        if citations:
            formatted_response += "\n\nCitazioni Rilevanti:\n"
            for citation in citations:
                formatted_response += f"- {citation['author']} ha scritto: '{citation['text']}'\n"
                
        return formatted_response