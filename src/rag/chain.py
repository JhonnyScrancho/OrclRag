from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from src.config import LLM_MODEL

def setup_rag_chain(retriever):
    """Configura la catena RAG."""
    llm = ChatOpenAI(
        model_name=LLM_MODEL,
        temperature=0,
        api_key=st.secrets["OPENAI_API_KEY"]
    )
    
    template = """Usa le seguenti informazioni per rispondere alla domanda. Se non conosci la risposta, di' semplicemente che non lo sai.

    Contesto: {context}
    
    Domanda: {query}
    
    Risposta:"""
    
    prompt = PromptTemplate(template=template, input_variables=["context", "query"])
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever.get_relevant_documents | format_docs, 
         "query": RunnablePassthrough()}
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    return chain