from typing import Dict, TypedDict
from langgraph.graph import END, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
import retriever

# Define the State
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    generation: str
    documents: list
    messages: list # Conversational memory

def retrieve(state):
    """
    Retrieve documents using the Advanced Hybrid + Graph Retriever.
    """
    question = state["question"]
    ret = retriever.AdvancedRetriever(top_k=5)
    documents = ret.invoke(question)
    return {"documents": documents, "question": question}

def generate(state):
    """
    Generate answer based on retrieved documents.
    """
    question = state["question"]
    documents = state["documents"]
    
    if not documents:
        return {"generation": "No encontré información relevante en la base de conocimiento para responder a tu pregunta.", "documents": documents, "question": question}
    
    # Format context
    context = "\n\n---\n\n".join([f"Fuente: {doc.metadata['title']}\nContenido: {doc.page_content}" for doc in documents])
    
    # Memory logic
    chat_history = state.get("messages", [])
    
    llm = ChatGroq(model="llama3-8b-8192")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente experto de AGV. Responde la pregunta del usuario basándote únicamente en el siguiente contexto. Si el contexto no contiene la respuesta, di que no tienes suficiente información. Sé conciso y directo.\n\nContexto:\n{context}\n\nHistorial de Chat:\n{history}"),
        ("human", "{question}")
    ])
    
    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])
    
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({"context": context, "history": history_str, "question": question})
    
    # Update messages
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=generation))
    
    return {"documents": documents, "question": question, "generation": generation, "messages": chat_history}

# Build the Graph
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve) 
workflow.add_node("generate", generate)

# Define edges
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Configure Memory
memory = MemorySaver()

# Compile
agent_app = workflow.compile(checkpointer=memory)
