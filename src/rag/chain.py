import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
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
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def get_response(query):
        try:
            # Get relevant documents
            docs = retriever.get_relevant_documents(query)
            
            # Format the context
            context = format_docs(docs)
            
            # Generate prompt
            formatted_prompt = prompt.format(context=context, query=query)
            
            # Get LLM response
            response = llm.invoke(formatted_prompt)
            
            # Extract the content from the message
            return response.content
            
        except Exception as e:
            st.error(f"Error in RAG chain: {str(e)}")
            return "Mi dispiace, c'è stato un errore nell'elaborazione della risposta."
    
    return get_response