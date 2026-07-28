import streamlit as st
import db
import retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import agent
import graph_store
import os

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.langchain import LangchainInstrumentor

@st.cache_resource
def init_telemetry():
    """Inicializa OpenTelemetry y exporta las trazas vía OTLP."""
    if not hasattr(trace.get_tracer_provider(), "add_span_processor"):
        resource = Resource.create({"service.name": "nexus-obsidian"})
        provider = TracerProvider(resource=resource)
        
        # Por defecto enviará trazas al localhost (ej. Jaeger o colector de OTel)
        otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")
        
        processor = BatchSpanProcessor(otlp_exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        
        # Instrumentar LangChain automáticamente
        LangchainInstrumentor().instrument()

init_telemetry()


# Configure page
st.set_page_config(page_title="Base de Conocimiento AGV", layout="wide", initial_sidebar_state="expanded")

# Initialize database
@st.cache_resource
def init_database():
    db.init_db()
    # Also preload the embedding model to avoid lag on first save/search
    db.get_model()
    # Preload retriever models
    retriever.get_all_docs_for_bm25()
    retriever.get_cross_encoder()

init_database()

@st.cache_resource
def get_llm():
    # IMPORTANT: You need to set GROQ_API_KEY in your environment variables
    # You can get a free key from https://console.groq.com/keys
    # For Streamlit Cloud, set this in the app's secrets.
    return ChatGroq(model="llama3-8b-8192")

def get_rag_chain():
    """Creates the RAG chain for answering questions."""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un asistente experto de AGV. Responde la pregunta del usuario basándote únicamente en el siguiente contexto. Si el contexto no contiene la respuesta, di que no tienes suficiente información. Sé conciso y directo.

Contexto:
{context}"""),
        ("human", "{question}")
    ])
    
    return prompt | llm | StrOutputParser()

# CSS for a more "Base de Conocimiento AGV" look
st.markdown("""
<style>
    .stApp {
        background-color: #1e1e1e;
        color: #d4d4d4;
    }
    .stTextInput>div>div>input {
        background-color: #2d2d2d;
        color: #d4d4d4;
    }
    .stTextArea>div>div>textarea {
        background-color: #2d2d2d;
        color: #d4d4d4;
        font-family: 'Courier New', Courier, monospace;
    }
    .note-card {
        padding: 10px;
        margin-bottom: 10px;
        background-color: #252526;
        border-radius: 5px;
        cursor: pointer;
        border: 1px solid #333;
    }
    .note-card:hover {
        background-color: #2a2d2e;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar (Navigation & Search) ---
with st.sidebar:
    st.image("Logo.png")
    st.title("🗂️ Chat con Documentos")
    st.divider()
    st.info("Esta es una demo de RAG Híbrido. Haz una pregunta sobre los documentos cargados.")
    # You can add other controls here in the future, like choosing a RAG strategy.

# --- Main Content Area ---
st.title("Bienvenido a la Base de Conocimiento AGV")

tab_chat, tab_graph = st.tabs(["💬 Chat (Agentic RAG)", "🕸️ Vista de Grafo (Graph View)"])

with tab_graph:
    st.subheader("Red de Conocimiento Global")
    if st.button("Actualizar Grafo"):
        graph_store.generate_graph_html()
    
    if os.path.exists("graph_view.html"):
        with open("graph_view.html", "r", encoding="utf-8") as f:
            source_code = f.read()
        st.components.v1.html(source_code, height=650, scrolling=True)
    else:
        st.info("El grafo aún no ha sido generado. Haz clic en actualizar o ingesta algunos documentos con enlaces.")

with tab_chat:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("¿Sobre qué quieres preguntar?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Pensando..."):
        # 1. Ejecutar Agentic RAG con LangGraph (ahora con Memoria)
        config = {"configurable": {"thread_id": "nexus-session-1"}}
        result = agent.agent_app.invoke({"question": prompt, "documents": []}, config=config)
        
        full_response = result["generation"]
        retrieved_docs = result["documents"]
        
        # Mostrar respuesta en la UI
        with st.chat_message("assistant"):
            st.markdown(full_response)
            
            # Mostrar fuentes si existen
            if retrieved_docs:
                with st.expander("Ver fuentes utilizadas"):
                    for doc in retrieved_docs:
                        st.info(f"**{doc.metadata.get('title', 'Desconocido')}** (Relevancia: {doc.metadata.get('relevance', 0):.2f})\n\n_{doc.page_content[:200]}..._")

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
