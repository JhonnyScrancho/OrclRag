import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from config import LLM_MODEL

def setup_rag_chain(retriever):
    """Configure the RAG chain."""
    llm = ChatOpenAI(
        model_name=LLM_MODEL,
        temperature=0,
        api_key=st.secrets["OPENAI_API_KEY"]
    )
    
    template = """Sei un assistente italiano esperto. Usa le seguenti informazioni per rispondere alla domanda in italiano. 
    Se non trovi informazioni pertinenti nel contesto, rispondi "Mi dispiace, non ho trovato informazioni rilevanti per rispondere alla tua domanda."

    Contesto fornito: {context}
    
    Domanda: {query}
    
    Risposta in italiano:"""
    
    prompt = PromptTemplate(template=template, input_variables=["context", "query"])
    
    def get_response(query_input):
        try:
            # Handle input dictionary
            if isinstance(query_input, dict):
                query = query_input.get("query", "")
            else:
                query = query_input
            
            # Get relevant documents
            docs = retriever.get_relevant_documents(query)
            
            # Log number of documents found
            st.write(f"Debug - Documenti trovati: {len(docs)}")
            
            # Format the context
            context = "\n\n".join(doc.page_content for doc in docs)
            
            # If no context found
            if not context.strip():
                return {"result": "Mi dispiace, non ho trovato informazioni rilevanti per rispondere alla tua domanda."}
            
            # Generate prompt
            formatted_prompt = prompt.format(context=context, query=query)
            
            # Create message
            messages = [HumanMessage(content=formatted_prompt)]
            
            # Get LLM response
            response = llm.invoke(messages)
            
            # Return the content
            result = response.content if hasattr(response, 'content') else str(response)
            return {"result": result}
            
        except Exception as e:
            st.error(f"Errore nella catena RAG: {str(e)}")
            return {"result": "Mi dispiace, c'è stato un errore nell'elaborazione della risposta."}
    
    return get_response