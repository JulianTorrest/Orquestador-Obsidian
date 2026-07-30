from typing import Dict, TypedDict
from langgraph.graph import END, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
import retriever
import db
from config import get_secret

# Define the State
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    generation: str
    documents: list
    messages: list # Conversational memory
    llm_choice: str  # openai | mistral

_LISTING_KEYWORDS = ["listado de", "lista de", "archivos", "ficheros", "notas", "documentos", "todos los archivos", "todos los ficheros", "todos los documentos", "todos los archivos"]
_COUNT_QUESTIONS = ["cuantos", "cuantas", "cuales son", "cuáles son", "cuántos", "cuántas", "cuales archivos", "cuáles archivos"]

def _is_listing_query(question: str) -> bool:
    q = question.lower()
    asks_for_list = any(k in q for k in _LISTING_KEYWORDS)
    asks_for_count = any(k in q for k in _COUNT_QUESTIONS) and any(k in q for k in ["archivos", "ficheros", "notas", "documentos"])
    return asks_for_list or asks_for_count

def retrieve(state):
    """
    Retrieve documents using the Advanced Hybrid + Graph Retriever.
    If the user asks for a list of files/notes, returns all titles from the DB.
    """
    question = state["question"]
    if _is_listing_query(question):
        notes_df = db.get_all_notes()
        if not notes_df.empty:
            titles = sorted(notes_df['title'].unique().tolist())
            titles_text = "\n".join(f"- {t}" for t in titles)
            doc = Document(
                page_content=f"Listado completo de archivos/notas disponibles ({len(titles)} total):\n\n{titles_text}",
                metadata={"title": "Listado de archivos"}
            )
            return {"documents": [doc], "question": question, "llm_choice": "openai"}
        else:
            return {"documents": [], "question": question, "llm_choice": "openai"}
    ret = retriever.AdvancedRetriever(top_k=5)
    documents = ret.invoke(question)
    return {"documents": documents, "question": question}

def select_llm(state):
    """
    LangGraph node that chooses the best LLM for the question.
    OpenAI is used for complex / reasoning prompts; Mistral as default.
    """
    question = state["question"].lower()
    has_openai = bool(get_secret("OPENAI_API_KEY"))
    has_mistral = bool(get_secret("MISTRAL_API_KEY"))

    # Default from .env
    default = get_secret("DEFAULT_LLM", "openai" if has_openai else "mistral").lower()

    # Heuristic: long or complex questions go to OpenAI
    complex_markers = ["explica", "compara", "analiza", "por que", "razona", "detalla", "lista", "resumen", "sintetiza"]
    is_complex = len(question.split()) > 15 or any(marker in question for marker in complex_markers)

    if is_complex and has_openai:
        choice = "openai"
    elif not has_openai and has_mistral:
        choice = "mistral"
    elif not has_mistral and has_openai:
        choice = "openai"
    else:
        choice = default

    return {"llm_choice": choice, "question": state["question"]}

def get_llm(choice: str):
    """Factory that returns the chosen LangChain chat model."""
    if choice == "mistral":
        return ChatMistralAI(
            model="mistral-large-latest",
            api_key=get_secret("MISTRAL_API_KEY"),
            temperature=0,
        )
    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=get_secret("OPENAI_API_KEY"),
        temperature=0,
    )

def _is_budget_error(err: Exception) -> bool:
    """Detects if an LLM error is related to quota/rate limit/insufficient credits."""
    text = str(err).lower()
    markers = ["quota", "rate limit", "exceeded", "insufficient", "credits", "billing", "payment", "429"]
    return any(m in text for m in markers)

def _build_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente experto de AGV. Responde la pregunta del usuario basándote únicamente en el siguiente contexto. Si el contexto no contiene la respuesta, di que no tienes suficiente información. Sé conciso y directo. Usa citas numéricas [1], [2], etc. para referenciar cada afirmación a la fuente correspondiente.\n\nContexto:\n{context}\n\nHistorial de Chat:\n{history}"),
        ("human", "{question}")
    ])
    return prompt | llm | StrOutputParser()

def generate(state):
    """
    Generate answer based on retrieved documents.
    Falls back to the other LLM if the selected one hits a budget/quota error.
    Short-circuits listing queries to avoid slow LLM calls.
    """
    question = state["question"]
    documents = state["documents"]
    
    if not documents:
        return {"generation": "No encontré información relevante en la base de conocimiento para responder a tu pregunta.", "documents": documents, "question": question}
    
    # Fast path: if the only document is a file listing, return it directly
    if len(documents) == 1 and documents[0].metadata.get("title") == "Listado de archivos":
        generation = documents[0].page_content
        chat_history = state.get("messages", [])
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=generation))
        return {"documents": documents, "question": question, "generation": generation, "messages": chat_history, "llm_choice": "direct"}
    
    # Format context with numbered sources
    context = "\n\n---\n\n".join([f"[{i+1}] Fuente: {doc.metadata['title']}\nContenido: {doc.page_content}" for i, doc in enumerate(documents)])
    
    # Memory logic
    chat_history = state.get("messages", [])
    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])
    
    llm_choice = state.get("llm_choice", "openai")
    
    def try_generate(choice):
        llm = get_llm(choice)
        chain = _build_chain(llm)
        return chain.invoke({"context": context, "history": history_str, "question": question})
    
    generation = None
    last_error = None
    for choice in [llm_choice, ("mistral" if llm_choice == "openai" else "openai")]:
        has_key = bool(get_secret("OPENAI_API_KEY")) if choice == "openai" else bool(get_secret("MISTRAL_API_KEY"))
        if not has_key:
            continue
        try:
            generation = try_generate(choice)
            llm_choice = choice
            break
        except Exception as e:
            last_error = (choice, e)
            if not _is_budget_error(e):
                break  # other errors are not recoverable by switching
    
    if generation is None:
        if last_error and _is_budget_error(last_error[1]):
            generation = f"Ambos LLM estan sin presupuesto o alcanzaron su limite. El ultimo error fue con {last_error[0]}: {last_error[1]}"
        elif last_error:
            generation = f"Error al generar la respuesta con {last_error[0]}: {last_error[1]}"
        else:
            generation = "No se pudo generar respuesta: ningun LLM esta configurado."
    
    # Update messages
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=generation))
    
    return {"documents": documents, "question": question, "generation": generation, "messages": chat_history, "llm_choice": llm_choice}

# Build the Graph
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve) 
workflow.add_node("select_llm", select_llm)
workflow.add_node("generate", generate)

# Define edges
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "select_llm")
workflow.add_edge("select_llm", "generate")
workflow.add_edge("generate", END)

# Configure Memory
memory = MemorySaver()

# Compile
agent_app = workflow.compile(checkpointer=memory)
