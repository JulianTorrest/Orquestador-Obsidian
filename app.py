import streamlit as st
import db
import graph_store
import os
import json
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import get_secret

# --- Note templates ---
TEMPLATES = {
    "Vacía": "",
    "Daily note": "---\ntags: [daily]\n---\n\n## Notas del día\n\n",
    "Proyecto": "---\ntags: [proyecto]\n---\n\n# Nombre del proyecto\n\n## Objetivos\n\n## Tareas\n",
    "Persona": "---\ntags: [persona]\n---\n\n# Nombre\n\n## Rol\n\n## Notas\n",
    "Reunión": "---\ntags: [reunión]\n---\n\n# Reunión: tema\n\n## Asistentes\n\n## Puntos clave\n\n## Acciones\n",
}

def init_telemetry():
    """Inicializa OpenTelemetry y exporta las trazas vía OTLP solo si esta activado."""
    if get_secret("LANGCHAIN_TRACING_V2", "false").lower() != "true":
        return
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.instrumentation.langchain import LangchainInstrumentor
    if not hasattr(trace.get_tracer_provider(), "add_span_processor"):
        resource = Resource.create({"service.name": "nexus-obsidian"})
        provider = TracerProvider(resource=resource)

        # Enviar trazas al endpoint de LangSmith configurado en .env
        endpoint = get_secret("LANGCHAIN_ENDPOINT", "http://localhost:4318") + "/v1/traces"
        otlp_exporter = OTLPSpanExporter(endpoint=endpoint)

        processor = BatchSpanProcessor(otlp_exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

        # Instrumentar LangChain automáticamente
        LangchainInstrumentor().instrument()

init_telemetry()


# Configure page
st.set_page_config(page_title="Base de Conocimiento AGV", layout="wide", initial_sidebar_state="expanded")

# Initialize database (always run to create missing tables)
db.init_db()

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", api_key=get_secret("OPENAI_API_KEY"))

def summarize_text(text, max_words=120):
    """Generates a concise TL;DR using the configured LLM."""
    if not text or not text.strip():
        return "No hay contenido para resumir."
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Resume el siguiente documento en un máximo de {max_words} palabras. Sé conciso y directo."),
        ("human", "{text}")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text[:6000], "max_words": max_words})

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
    st.title("Chat con Documentos")
    st.divider()
    st.info("Esta es una demo de RAG Híbrido. Haz una pregunta sobre los documentos cargados.")

    pinned = db.get_pinned_notes()
    if pinned:
        st.markdown("### Favoritos")
        for p in pinned:
            st.markdown(f"- `{p}`")
    # You can add other controls here in the future, like choosing a RAG strategy.

# --- Main Content Area ---
st.title("Bienvenido a la Base de Conocimiento AGV")

tab_graph, tab_library, tab_chat, tab_dashboard, tab_compare, tab_timeline = st.tabs([
    "Vista de Grafo (Graph View)", "Biblioteca", "Chat (Agentic RAG)", "Dashboard", "Comparar", "Cronologia"
])

with tab_graph:
    st.subheader("Red de Conocimiento Global")
    if st.button("Actualizar Grafo"):
        try:
            # Reconstruir grafo a partir de los documentos ya indexados
            with st.spinner("Reconstruyendo grafo..."):
                all_notes = db.get_all_notes_full_content()
                graph_store.rebuild_graph(all_notes)
                graph_store.generate_graph_html()
            st.success("Grafo generado.")
            st.rerun()
        except Exception as e:
            st.error(f"Error al generar el grafo: {e}")
    
    if os.path.exists("graph_view.html"):
        try:
            with open("graph_view.html", "r", encoding="utf-8") as f:
                source_code = f.read()
            if source_code.strip():
                st.components.v1.html(source_code, height=650, scrolling=True)
            else:
                st.info("El archivo del grafo está vacío.")
        except Exception as e:
            st.error(f"No se pudo mostrar el grafo: {e}")
    else:
        st.info("El grafo aún no ha sido generado. Haz clic en actualizar o ingesta algunos documentos con enlaces.")

with tab_library:
    st.subheader("Biblioteca de Conocimiento")
    if st.button("Recargar biblioteca"):
        st.cache_data.clear()
        st.rerun()
    search_query = st.text_input("Buscar notas", placeholder="Escribe una palabra o pregunta...")
    col1, col2 = st.columns([1, 3])
    with col1:
        search_mode = st.selectbox("Modo", ["Semántica", "Palabra clave", "Avanzada"])
    with col2:
        top_n = st.slider("Máximo resultados", 5, 50, 10)
    
    if search_query:
        if search_mode == "Semántica":
            results = db.search_notes(search_query, limit=top_n)
        elif search_mode == "Palabra clave":
            results = db.keyword_search(search_query, limit=top_n)
        else:
            results = db.advanced_search(search_query, limit=top_n)
    else:
        results = db.get_all_notes()
    
    if search_mode == "Avanzada":
        with st.expander("Sintaxis de búsqueda avanzada"):
            st.markdown("""
            - `tag:legal` — filtra por tag
            - `title:empresa` — busca en título
            - `content:reglamento` — busca en contenido
            - `proyecto AND legal` — ambas palabras
            - `"impuesto nacional"` — frase exacta
            - `NOT privado` — excluye
            - `regex:\\d{4}` — expresión regular
            """)
    
    left, right = st.columns([1, 2])
    
    with left:
        st.markdown("### Importar / Crear")

        # Import markdown file
        uploaded = st.file_uploader("Importar Markdown", type=["md"])
        if uploaded is not None:
            file_name = uploaded.name
            raw = uploaded.read().decode("utf-8", errors="ignore")
            if st.button("Indexar nota importada"):
                db.update_note_content(file_name, raw)
                db.save_metadata(file_name, "", "{}")
                st.success(f"Nota {file_name} importada.")
                st.rerun()

        # Bulk import ZIP
        zip_upload = st.file_uploader("Importar ZIP de Markdowns", type=["zip"])
        if zip_upload is not None:
            if st.button("Indexar ZIP"):
                import zipfile
                with zipfile.ZipFile(zip_upload) as z:
                    for name in z.namelist():
                        if name.endswith(".md"):
                            raw = z.read(name).decode("utf-8", errors="ignore")
                            db.update_note_content(name, raw)
                            db.save_metadata(name, "", "{}")
                st.success("ZIP importado.")
                st.rerun()

        st.divider()

        # Bulk export ZIP
        if st.button("Exportar todo a ZIP"):
            import zipfile
            import io
            buffer = io.BytesIO()
            all_notes = db.get_all_notes_full_content()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
                for note in all_notes:
                    z.writestr(note['title'], note['content'] or "")
            st.download_button(
                "Descargar ZIP",
                data=buffer.getvalue(),
                file_name="obsidian_backup.zip",
                mime="application/zip"
            )

        # Create note from template
        new_title = st.text_input("Nueva nota: título")
        template = st.selectbox("Plantilla", list(TEMPLATES.keys()))
        if new_title and st.button("Crear nota"):
            db.update_note_content(new_title, TEMPLATES[template])
            db.save_metadata(new_title, "", "{}")
            st.success(f"Nota {new_title} creada.")
            st.rerun()
        
        st.divider()
        
        # Tags filter
        all_tags = db.get_all_tags()
        selected_tags = st.multiselect("Filtrar por tags", all_tags)
        
        if results.empty:
            st.info("No se encontraron notas.")
            selected_title = None
        else:
            titles = results['title'].unique().tolist()
            
            if selected_tags:
                allowed = set(db.get_all_notes()['title'].unique().tolist())
                for tag in selected_tags:
                    allowed &= set(db.get_titles_by_tag(tag))
                titles = [t for t in titles if t in allowed]
            
            if not titles:
                st.info("Ninguna nota coincide con los filtros.")
                selected_title = None
            else:
                st.caption(f"{len(titles)} notas encontradas")
                selected_title = st.selectbox("Seleccionar nota", titles, key="lib_select")
    
    with right:
        if selected_title:
            col_title, col_pin = st.columns([4, 1])
            with col_title:
                st.markdown(f"## {selected_title}")
            with col_pin:
                if db.is_pinned(selected_title):
                    if st.button("Desfijar", key=f"unpin_{selected_title}"):
                        db.unpin_note(selected_title)
                        st.rerun()
                else:
                    if st.button("Fijar", key=f"pin_{selected_title}"):
                        db.pin_note(selected_title)
                        st.rerun()
            content = db.get_note_by_title(selected_title)
            metadata = db.get_metadata(selected_title)

            if metadata['tags']:
                st.caption(f"Tags: {metadata['tags']}")
            
            view_tab, edit_tab, meta_tab = st.tabs(["Vista", "Editar", "Metadatos"])
            
            with view_tab:
                rendered = db.render_content_with_embeds(content) if content else "*Sin contenido*"
                st.markdown(rendered)

                st.divider()
                st.markdown("### Preguntas sugeridas")
                if st.button("Generar preguntas", key=f"gen_q_{selected_title}"):
                    with st.spinner("Generando..."):
                        prompt = ChatPromptTemplate.from_messages([
                            ("system", "Genera exactamente 3 preguntas cortas que un usuario haría sobre este documento. Devuélvelas numeradas."),
                            ("human", "{text}")
                        ])
                        chain = prompt | get_llm() | StrOutputParser()
                        questions = chain.invoke({"text": content[:4000]})
                        st.session_state[f"questions_{selected_title}"] = questions
                if f"questions_{selected_title}" in st.session_state:
                    st.info(st.session_state[f"questions_{selected_title}"])

                st.divider()
                st.markdown("### Resumen (TL;DR)")
                saved_summary = db.get_summary(selected_title)
                if saved_summary:
                    st.info(saved_summary)
                    if st.button("Regenerar resumen", key=f"resum_{selected_title}"):
                        summary = summarize_text(content)
                        db.save_summary(selected_title, summary)
                        st.rerun()
                else:
                    if st.button("Generar resumen", key=f"gen_resum_{selected_title}"):
                        summary = summarize_text(content)
                        db.save_summary(selected_title, summary)
                        st.rerun()
                
                st.divider()
                st.markdown("### Grafo local")
                degree = st.slider("Grado de vecindad", 1, 3, 1, key=f"deg_{selected_title}")
                if st.button("Ver grafo local", key=f"local_graph_{selected_title}"):
                    try:
                        graph_store.generate_local_graph_html(selected_title, degree=degree)
                        with open("local_graph.html", "r", encoding="utf-8") as f:
                            local_html = f.read()
                        st.components.v1.html(local_html, height=400, scrolling=True)
                    except Exception as e:
                        st.error(f"No se pudo generar el grafo local: {e}")
                
                st.divider()
                st.markdown("### Conexiones")
                conn_col1, conn_col2 = st.columns(2)
                with conn_col1:
                    st.markdown("**Enlaces salientes**")
                    out_links = db.get_forward_links(selected_title)
                    if out_links:
                        for ol in out_links:
                            st.markdown(f"- `{ol}`")
                    else:
                        st.info("Sin enlaces salientes.")

                    st.markdown("**Backlinks (enlaces entrantes)**")
                    backlinks = db.get_backlinks(selected_title)
                    if backlinks:
                        for bl in backlinks:
                            st.markdown(f"- `{bl}`")
                    else:
                        st.info("Ninguna otra nota enlaza o menciona esta nota.")
                with conn_col2:
                    st.markdown("**Menciones no enlazadas**")
                    unlinked = db.get_unlinked_mentions(selected_title)
                    if unlinked:
                        for ul in unlinked:
                            st.markdown(f"- `{ul}`")
                    else:
                        st.info("Sin menciones no enlazadas.")

                    st.markdown("**Notas sugeridas**")
                    try:
                        similar = db.get_similar_notes(selected_title)
                        if similar:
                            for sim in similar:
                                st.markdown(f"- `{sim}`")
                        else:
                            st.info("Sin notas similares.")
                    except Exception:
                        st.info("No se pudo calcular similitud.")

                st.divider()
                st.markdown("### Entidades y referencias")
                entities = db.extract_entities(selected_title)
                if entities:
                    for key, items in entities.items():
                        if items:
                            with st.expander(f"{key.capitalize()} ({len(items)})"):
                                st.markdown(", ".join([f"`{e}`" for e in items]))
                else:
                    st.info("No se encontraron entidades.")
            
            with edit_tab:
                new_content = st.text_area(
                    "Contenido Markdown",
                    value=content,
                    height=400,
                    key=f"edit_{selected_title}"
                )
                if st.button("Guardar cambios", key=f"save_{selected_title}"):
                    db.update_note_content(selected_title, new_content)
                    st.success("Nota guardada y reindexada.")
            
            with meta_tab:
                if metadata['tags']:
                    st.markdown("**Tags:**")
                    for tag in [t.strip() for t in metadata['tags'].split(",") if t.strip()]:
                        st.markdown(f"- `{tag}`")
                else:
                    st.info("Sin tags.")
                st.markdown("**YAML frontmatter:**")
                st.json(metadata['frontmatter'] if metadata['frontmatter'] != "{}" else "{}")
                
                st.divider()
                st.markdown("**Tareas / Checklist**")
                tasks = db.get_tasks(selected_title)
                if not tasks.empty:
                    for _, row in tasks.iterrows():
                        col_a, col_b = st.columns([0.1, 0.9])
                        with col_a:
                            new_status = st.checkbox("", value=bool(row['completed']), key=f"task_{row['id']}")
                        with col_b:
                            st.markdown(f"{row['task_text']}")
                        if new_status != bool(row['completed']):
                            db.update_task_status(row['id'], new_status)
                            st.rerun()
                
                new_task = st.text_input("Nueva tarea", key=f"new_task_{selected_title}")
                if new_task and st.button("Agregar tarea", key=f"add_task_{selected_title}"):
                    db.add_task(selected_title, new_task)
                    st.rerun()
                
                st.divider()
                st.markdown("**Exportar**")
                export_name = selected_title if selected_title.endswith(".md") else f"{selected_title}.md"
                st.download_button(
                    "Descargar Markdown",
                    data=content,
                    file_name=export_name,
                    mime="text/markdown",
                    key=f"export_{selected_title}"
                )

with tab_chat:
    import agent  # lazy: only load on chat tab
    # Chat session persistence
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if "chat_session_name" not in st.session_state:
        st.session_state.chat_session_name = "Nueva conversación"
    if "messages" not in st.session_state:
        st.session_state.messages = []

    sessions = db.list_chat_sessions()
    col_session, col_name = st.columns([2, 3])
    with col_session:
        session_options = {row['session_id']: (row['name'] or row['session_id']) for _, row in sessions.iterrows()}
        session_options["Nueva conversación"] = "Nueva conversación"
        selected_session = st.selectbox(
            "Conversación",
            list(session_options.keys()),
            format_func=lambda x: session_options.get(x, x),
            index=list(session_options.keys()).index(st.session_state.chat_session_id) if st.session_state.chat_session_id in session_options else len(session_options) - 1
        )
        if selected_session != st.session_state.chat_session_id and selected_session != "Nueva conversación":
            st.session_state.chat_session_id = selected_session
            st.session_state.chat_session_name = session_options[selected_session]
            st.session_state.messages = db.load_chat_session(selected_session)
            st.rerun()
        elif selected_session == "Nueva conversación":
            if st.button("Crear nueva"):
                st.session_state.chat_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.session_state.chat_session_name = "Nueva conversación"
                st.session_state.messages = []
                st.rerun()
    with col_name:
        new_name = st.text_input("Nombre de la conversación", value=st.session_state.chat_session_name)
        if new_name and new_name != st.session_state.chat_session_name:
            st.session_state.chat_session_name = new_name

    if st.button("Guardar conversación"):
        db.save_chat_session(
            st.session_state.chat_session_id,
            st.session_state.chat_session_name,
            st.session_state.messages
        )
        st.success("Conversación guardada.")

    if st.session_state.messages:
        md_lines = []
        for m in st.session_state.messages:
            md_lines.append(f"**{m['role'].capitalize()}:** {m['content']}\n")
        md_export = "\n".join(md_lines)
        st.download_button(
            "Exportar conversación",
            data=md_export,
            file_name=f"chat_{st.session_state.chat_session_name}.md",
            mime="text/markdown"
        )
    else:
        st.write("")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("¿Sobre qué quieres preguntar?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Pensando..."):
        config = {"configurable": {"thread_id": st.session_state.chat_session_id}}
        result = agent.agent_app.invoke({"question": prompt, "documents": []}, config=config)
        full_response = result["generation"]
        retrieved_docs = result["documents"]

        with st.chat_message("assistant"):
            st.markdown(full_response)

            if retrieved_docs:
                with st.expander("Ver fuentes utilizadas"):
                    for i, doc in enumerate(retrieved_docs, 1):
                        st.info(f"**[{i}] {doc.metadata.get('title', 'Desconocido')}** (Relevancia: {doc.metadata.get('relevance', 0):.2f})\n\n_{doc.page_content[:200]}..._")

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        db.save_chat_session(
            st.session_state.chat_session_id,
            st.session_state.chat_session_name,
            st.session_state.messages
        )

with tab_dashboard:
    st.subheader("Dashboard")
    kpis = db.get_kpi_stats()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Notas", kpis['total_titles'])
    c2.metric("Chunks", kpis['total_chunks'])
    c3.metric("Tags", kpis['total_tags'])
    c4.metric("Tareas", kpis['total_tasks'])
    c5.metric("Chats", kpis['total_sessions'])

    st.divider()
    st.markdown("### Guia del Sistema")
    st.markdown("Esta aplicacion es un asistente conversacional sobre una base de conocimiento privada. El objetivo es responder preguntas usando solo la informacion indexada de tus notas y documentos.")

    with st.expander("1. Arquitectura general", expanded=True):
        st.markdown("El flujo completo tiene 4 capas principales: **Ingesta**, **Almacenamiento**, **Recuperacion (RAG)** y **Generacion / Orquestacion**.")
        st.code("""
+----------------+     +----------------+     +---------------------+     +------------------+
|   Fuentes      |     |   DuckDB       |     |   RAG Hibrido       |     |   LangGraph      |
|   (.md, .pdf,  | --> |   obsidian.    | --> |   (embeddings +     | --> |   ChatOpenAI     |
|    .docx, .xlsx)|     |   duckdb       |     |    BM25 + grafo +   |     |   respuesta      |
|                |     |   con vectors  |     |    cross-encoder)   |     |   con citas      |
+----------------+     +----------------+     +---------------------+     +------------------+
        ^                                                                      |
        |                                                                      v
        |                                                              +----------------+
        +--------------------------------------------------------------|   Respuesta    |
                                                                       |   final        |
                                                                       +----------------+
        """, language="text")
        st.markdown("**Ingesta**: `ingest.py` lee archivos, los fragmenta, genera embeddings y los guarda en DuckDB.")
        st.markdown("**Almacenamiento**: DuckDB guarda notas, chunks, embeddings `FLOAT[]`, metadatos, tareas y sesiones de chat.")
        st.markdown("**Recuperacion**: `retriever.py` combina busqueda semantica, BM25, expansion por vecinos en el grafo y re-ranking con cross-encoder.")
        st.markdown("**Orquestacion**: `agent.py` usa LangGraph con estados `retrieve` y `generate` para decidir cuando buscar y cuando responder.")

    with st.expander("2. LangChain"):
        st.markdown("LangChain es la capa de abstraccion sobre modelos de lenguaje. Aca se encarga de:")
        st.markdown("- `ChatOpenAI(model='gpt-4o-mini')`: genera respuestas basadas en contexto.")
        st.markdown("- `ChatPromptTemplate`: junta el system prompt, el historial de chat y el contexto recuperado.")
        st.markdown("- Cadenas de parsing: toma el output del LLM y lo convierte en markdown legible.")
        st.markdown("Todo se conecta en `agent.py` para que el modelo reciba el contexto exacto que necesita.")

    with st.expander("3. LangGraph"):
        st.markdown("LangGraph modela el agente como un grafo de estados dirigido. En esta aplicacion los nodos principales son:")
        st.markdown("- **retrieve**: recibe la pregunta, busca documentos relevantes y los guarda en el estado.")
        st.markdown("- **select_llm**: elige el mejor LLM entre **OpenAI gpt-4o-mini** y **Mistral mistral-large-latest** segun la pregunta y disponibilidad de API keys.")
        st.markdown("- **generate**: toma los documentos y la pregunta, llama al LLM seleccionado y produce la respuesta con citas numericas [1], [2], etc. Si el LLM seleccionado falla por falta de presupuesto o cuota, intenta automaticamente con el otro LLM.")
        st.code("""
    +-----------+    +-----------+    +-----------+    +-----------+    +-----------+
    |  START    | -> |  retrieve | -> | select_llm| -> |  generate | -> | fallback  | -> END
    |  question |    |  (buscar) |    | (elegir)  |    | (responder)|    | (presup.) |
    +-----------+    +-----------+    +-----------+    +-----------+    +-----------+
        """, language="text")
        st.markdown("El nodo `select_llm` usa heuristicas: preguntas largas, que piden explicar, comparar, analizar o razonar se envian a OpenAI; otras pueden ir a Mistral. El default se controla con `DEFAULT_LLM` en `.env`.")
        st.markdown("Si el LLM elegido responde con error de cuota, rate limit o presupuesto agotado, el sistema prueba el otro proveedor antes de rendirse.")

    with st.expander("4. RAG Hibrido"):
        st.markdown("El RAG (Retrieval-Augmented Generation) de esta aplicacion no se basa solo en una busqueda vectorial, sino en varias tecnicas combinadas para mejorar la relevancia.")
        st.code("""
Pregunta del usuario
       |
       v
+-----------------------------+
| 1. Embedding de la pregunta |  all-MiniLM-L6-v2
+-----------------------------+
       |
       v
+-----------------------------+
| 2. Busqueda semantica       |  cosine similarity en DuckDB
+-----------------------------+
       |
       v
+-----------------------------+
| 3. Busqueda BM25            |  rank_bm25 sobre contenido
+-----------------------------+
       |
       v
+-----------------------------+
| 4. Expansion por grafo      |  vecinos [[...]] en NetworkX
+-----------------------------+
       |
       v
+-----------------------------+
| 5. Re-ranking cross-encoder |  ms-marco-MiniLM-L-6-v2
+-----------------------------+
       |
       v
+-----------------------------+
| 6. Top-k chunks             |  se envian al LLM
+-----------------------------+
        """, language="text")
        st.markdown("**Busqueda semantica**: convierte pregunta y documentos a vectores y busca por similitud de coseno.")
        st.markdown("**BM25**: busqueda lexica por terminos exactos para palabras clave y nombres propios.")
        st.markdown("**Expansion por grafo**: si una nota es relevante, tambien se incluyen sus vecinos conectados por enlaces `[[...]]`.")
        st.markdown("**Cross-encoder**: re-ordena los candidatos dandole un score de relevancia a cada par pregunta-documento.")

    with st.expander("5. Base de datos vectorial"):
        st.markdown("La base de datos es **DuckDB**, un motor OLAP embebido. Se eligio porque:")
        st.markdown("- No requiere servidor externo.")
        st.markdown("- Soporta arrays `FLOAT[]` para embeddings.")
        st.markdown("- Permite SQL estandar con operaciones vectoriales.")
        st.markdown("- Es rapido y ligero para un solo usuario.")
        st.markdown("Tablas principales:")
        st.markdown("- `notes`: titulos, contenido, chunks y embeddings.")
        st.markdown("- `metadata`: tags, fechas, entidades extraidas.")
        st.markdown("- `note_links`: enlaces `[[...]]` detectados.")
        st.markdown("- `tasks`: tareas y checklists.")
        st.markdown("- `chat_sessions`: historial de conversaciones.")

    with st.expander("6. Pipeline de una pregunta"):
        st.code("""
Usuario escribe pregunta
        |
        v
LangGraph invoca retrieve
        |
        v
Retriever consulta DuckDB (semantic + BM25 + grafo)
        |
        v
Cross-encoder reordena los chunks
        |
        v
LangGraph invoca generate con los top-k chunks
        |
        v
ChatOpenAI responde con citas [1], [2]...
        |
        v
Chat muestra respuesta y fuentes resaltadas
        """, language="text")

    with st.expander("7. Recomendaciones"):
        st.markdown("Para sacar el maximo provecho:")
        st.markdown("- Usa enlaces `[[Nombre de Nota]]` para conectar ideas; el grafo mejora la recuperacion.")
        st.markdown("- Etiqueta las notas con YAML frontmatter (`tags: [proyecto, reunion]`).")
        st.markdown("- Indexa documentos en lotes con `ingest.py` para no saturar la memoria.")
        st.markdown("- Exporta tus chats y tareas periodicamente a Markdown para respaldo.")
        st.markdown("- Usa el Dashboard para revisar metricas, clusters y tareas pendientes.")

    st.divider()
    st.markdown("### Alertas por cambios en carpeta")
    folder = st.text_input("Carpeta a monitorear", value=get_secret("DATA_FOLDER", "data"), key="alert_dashboard_folder")
    if st.button("Escanear carpeta", key="alert_dashboard_btn"):
        scan = db.scan_for_changes(folder)
        if "error" in scan:
            st.error(scan["error"])
        else:
            st.markdown(f"**Indexados:** {scan['total_indexed']} | **En disco:** {scan['total_on_disk']}")
            if scan['new_files']:
                st.warning(f"Archivos nuevos no indexados ({len(scan['new_files'])})")
                for f in scan['new_files']:
                    st.markdown(f"- `{f}`")
            else:
                st.success("No hay archivos nuevos.")
            if scan['missing_files']:
                st.warning(f"Archivos indexados que ya no existen ({len(scan['missing_files'])})")
                for f in scan['missing_files']:
                    st.markdown(f"- `{f}`")

    st.divider()
    st.markdown("### Notas recientemente actualizadas")
    if not kpis['recent'].empty:
        st.dataframe(kpis['recent'])
    else:
        st.info("No hay notas aún.")
    
    st.divider()
    st.markdown("### Métricas del grafo")
    try:
        g_metrics = graph_store.get_graph_metrics()
        if g_metrics:
            gm1, gm2, gm3, gm4 = st.columns(4)
            gm1.metric("Nodos", g_metrics.get('nodes', 0))
            gm2.metric("Aristas", g_metrics.get('edges', 0))
            gm3.metric("Clusters", g_metrics.get('clusters', 0))
            gm4.metric("Cluster más grande", g_metrics.get('largest_cluster_size', 0))
            if g_metrics.get('top_connected'):
                st.markdown("**Notas más conectadas**")
                for node, degree in g_metrics['top_connected'][:5]:
                    st.markdown(f"- `{node}`: {degree} conexiones")
        else:
            st.info("Grafo vacío. Genera el grafo desde la pestaña Vista de Grafo.")
    except Exception as e:
        st.error(f"No se pudieron cargar métricas del grafo: {e}")

    st.divider()
    st.markdown("### Clustering temático")
    try:
        clusters = db.cluster_notes(similarity_threshold=0.75)
        if clusters:
            for c in clusters:
                with st.expander(f"Tema: {c['label']} ({len(c['notes'])} notas)"):
                    for note in c['notes']:
                        st.markdown(f"- `{note}`")
        else:
            st.info("No hay suficientes notas para agrupar.")
    except Exception as e:
        st.error(f"No se pudo calcular clustering: {e}")

    st.divider()
    st.markdown("### Mapa de calor de conexiones")
    try:
        g_metrics = graph_store.get_graph_metrics()
        if g_metrics.get('top_connected'):
            data = {node: degree for node, degree in g_metrics['top_connected'][:15]}
            st.bar_chart(data)
        else:
            st.info("No hay datos de grafo para el mapa de calor.")
    except Exception as e:
        st.error(f"No se pudo cargar el mapa de calor: {e}")

    st.divider()
    st.markdown("### Tareas abiertas")
    all_tasks = db.get_all_tasks()
    if not all_tasks.empty:
        st.dataframe(all_tasks[all_tasks['completed'] == False])
    else:
        st.info("No hay tareas registradas.")

with tab_compare:
    st.subheader("Comparar dos notas")
    all_titles = db.get_all_notes()['title'].unique().tolist()
    if len(all_titles) < 2:
        st.info("Necesitas al menos dos notas para comparar.")
    else:
        a, b = st.columns(2)
        with a:
            title_a = st.selectbox("Nota A", all_titles, key="cmp_a")
        with b:
            title_b = st.selectbox("Nota B", [t for t in all_titles if t != title_a], key="cmp_b")
        if st.button("Comparar"):
            result = db.compare_notes(title_a, title_b)
            if not result:
                st.error("No se pudieron cargar las notas.")
            else:
                st.markdown(f"**Palabras Nota A:** {result['words_a']} | **Nota B:** {result['words_b']}")
                st.markdown(f"**Palabras compartidas:** {result['shared_words']}")
                st.markdown(f"**Similitud semántica:** {result['similarity']:.3f}")
                if result['shared_sample']:
                    with st.expander("Palabras comunes (muestra)"):
                        st.markdown(", ".join([f"`{w}`" for w in result['shared_sample']]))

with tab_timeline:
    st.subheader("Cronologia de eventos")
    events = db.build_timeline()
    if events:
        st.markdown("### Filtros")
        filter_title = st.text_input("Filtrar por título de nota", "")
        filter_date = st.text_input("Filtrar por fecha", "")

        filtered = [
            ev for ev in events
            if filter_title.lower() in ev['title'].lower()
            and filter_date in ev['date']
        ]

        if filtered:
            st.markdown(f"**Mostrando {len(filtered)} eventos**")
            for ev in filtered:
                st.markdown(f"**{ev['date']}** — *{ev['title']}*")
                st.caption(ev['snippet'])
        else:
            st.info("Ningún evento coincide con los filtros.")
    else:
        st.info("No se encontraron fechas en los documentos.")
