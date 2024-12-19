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
    
    template = """Use the following information to answer the question. If you don't know the answer, simply say you don't know.

    Context: {context}
    
    Question: {query}
    
    Answer:"""
    
    prompt = PromptTemplate(template=template, input_variables=["context", "query"])
    
    def get_response(query_input):
        try:
            # Handle input dictionary
            if isinstance(query_input, dict):
                query = query_input.get("query", "")
            else:
                query = query_input
            
            # Log the query
            st.write("Debug - Query:", query)
            
            # Get relevant documents
            docs = retriever.get_relevant_documents(query)
            
            # Format the context
            context = "\n\n".join(doc.page_content for doc in docs)
            
            # Log the context
            st.write("Debug - Context length:", len(context))
            
            # Generate prompt
            formatted_prompt = prompt.format(context=context, query=query)
            
            # Create message
            messages = [HumanMessage(content=formatted_prompt)]
            
            # Get LLM response
            response = llm.invoke(messages)
            
            # Log the response
            st.write("Debug - Response type:", type(response))
            
            # Return the content
            result = response.content if hasattr(response, 'content') else str(response)
            return {"result": result}
            
        except Exception as e:
            st.error(f"Error in RAG chain: {str(e)}")
            st.write("Debug - Error details:", str(e))
            return {"result": "Mi dispiace, c'è stato un errore nell'elaborazione della risposta."}
    
    return get_response