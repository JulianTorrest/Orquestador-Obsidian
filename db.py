import duckdb
import os
import uuid
import pandas as pd
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Lazy import: SentenceTransformer is heavy; only load it when needed.
SentenceTransformer = None

# Initialize the embedding model
# all-MiniLM-L6-v2 is a good balance between size and performance for semantic search
MODEL_NAME = 'all-MiniLM-L6-v2'
model = None

# --- Caching Logic ---
# This function will decide whether to use Streamlit's cache or a dummy decorator
def get_caching_decorators():
    try:
        from streamlit.runtime.caching import cache_data, cache_resource
        return cache_data, cache_resource
    except ImportError:
        # If Streamlit is not installed or not in a streamlit context, return dummy decorators
        return lambda func, **kwargs: func, lambda func, **kwargs: func

st_cache_data, st_cache_resource = get_caching_decorators()

@st_cache_resource(show_spinner=False)
def get_model():
    global model, SentenceTransformer
    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as ST
        SentenceTransformer = ST
    if model is None:
        model = SentenceTransformer(MODEL_NAME)
    return model

# Database path
DB_PATH = 'obsidian.duckdb'

def get_connection():
    return duckdb.connect(DB_PATH)

def init_db(conn=None):
    """Initializes the database schema."""
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id VARCHAR PRIMARY KEY,
            title VARCHAR NOT NULL, -- Source file name
            content VARCHAR,
            chunk_index INTEGER, -- Order of the chunk in the document
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            embedding FLOAT[]
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS note_metadata (
            title VARCHAR PRIMARY KEY,
            tags VARCHAR,
            frontmatter VARCHAR,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS note_summaries (
            title VARCHAR PRIMARY KEY,
            summary VARCHAR,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id VARCHAR PRIMARY KEY,
            name VARCHAR,
            messages VARCHAR,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS note_tasks (
            id VARCHAR PRIMARY KEY,
            note_title VARCHAR,
            task_text VARCHAR,
            completed BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS note_pins (
            title VARCHAR PRIMARY KEY,
            pinned BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    if close_conn:
        conn.close()

def get_embedding(text):
    """Generates an embedding for the given text."""
    embed_model = get_model()
    return embed_model.encode(text).tolist()

def get_embeddings(texts):
    """Generates embeddings for a list of texts in a batch."""
    if not texts:
        return []
    embed_model = get_model()
    return embed_model.encode(texts, show_progress_bar=True, batch_size=64).tolist()

def insert_chunk(conn, chunk_id, source_file, content, chunk_index):
    """Inserts a document chunk and its embedding using an existing connection."""
    embedding = get_embedding(content)
    current_time = datetime.now()
    
    # Para evitar el "Binder Error" de DuckDB con CURRENT_TIMESTAMP en la cláusula ON CONFLICT,
    # generamos la marca de tiempo en Python y la pasamos como un parámetro.
    # Esto elimina cualquier ambigüedad para el analizador de SQL.
    conn.execute(
        """
        INSERT INTO notes (id, title, content, chunk_index, embedding) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (id) DO UPDATE SET
            content = excluded.content,
            embedding = excluded.embedding,
            chunk_index = excluded.chunk_index,
            updated_at = ?
        """, (chunk_id, source_file, content, chunk_index, embedding, current_time))

def insert_chunks(conn, chunks):
    """Inserts multiple document chunks and their embeddings in a batch."""
    if not chunks:
        return

    texts = [content for _, _, content, _ in chunks]
    embeddings = get_embeddings(texts)
    current_time = datetime.now()

    data = [
        (chunk_id, source_file, content, chunk_index, embedding, current_time)
        for (chunk_id, source_file, content, chunk_index), embedding in zip(chunks, embeddings)
    ]

    conn.executemany(
        """
        INSERT INTO notes (id, title, content, chunk_index, embedding, updated_at) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (id) DO UPDATE SET
            content = excluded.content,
            embedding = excluded.embedding,
            chunk_index = excluded.chunk_index,
            updated_at = excluded.updated_at
        """, data)

@st_cache_data(ttl=10)
def get_all_notes():
    """Returns all notes ordered by updated_at."""
    conn = get_connection()
    df = conn.execute("SELECT id, title, updated_at FROM notes ORDER BY updated_at DESC").fetchdf()
    conn.close()
    return df

def get_all_notes_full_content():
    """Returns all notes with their full content for indexing."""
    conn = get_connection()
    # Using fetchall() to get a list of tuples, then converting to a list of dicts
    results = conn.execute("SELECT id, title, content FROM notes").fetchall()
    conn.close()
    notes = [{'id': r[0], 'title': r[1], 'content': r[2]} for r in results]
    return notes

def get_note(note_id):
    """Retrieves a single note by ID."""
    conn = get_connection()
    result = conn.execute("SELECT id, title, content, updated_at FROM notes WHERE id = ?", (note_id,)).fetchone()
    conn.close()
    
    if result:
        return {
            'id': result[0],
            'title': result[1],
            'content': result[2],
            'updated_at': result[3]
        }
    return None

def get_note_by_title(title):
    """Reconstructs the full content of a note from its chunks."""
    conn = get_connection()
    results = conn.execute(
        "SELECT content, chunk_index FROM notes WHERE title = ? ORDER BY chunk_index ASC",
        (title,)
    ).fetchall()
    conn.close()
    if not results:
        return ""
    return "\n\n".join([r[0] for r in results])

def render_content_with_embeds(content, depth=0, max_depth=2):
    """Resolves ![[Note]] transclusions by embedding the referenced note content."""
    import re
    if depth > max_depth:
        return content
    pattern = r'!\[\[(.*?)\]\]'
    
    def repl(match):
        ref_title = match.group(1).strip()
        try:
            embedded = get_note_by_title(ref_title)
            rendered = render_content_with_embeds(embedded, depth + 1, max_depth)
            return f"\n\n---\n\n**Incrustado: `{ref_title}`**\n\n{rendered}\n\n---\n\n"
        except Exception:
            return f"_(No se pudo incrustar `{ref_title}`)_"
    
    return re.sub(pattern, repl, content)

def get_backlinks(title):
    """Returns titles of notes that mention the given title."""
    conn = get_connection()
    explicit_link = f"%[[{title}]]%"
    mention = f"%{title}%"
    df = conn.execute(
        """
        SELECT DISTINCT title
        FROM notes
        WHERE (content ILIKE ? OR content ILIKE ?) AND title <> ?
        """,
        (explicit_link, mention, title)
    ).fetchdf()
    conn.close()
    return df['title'].unique().tolist()

def get_forward_links(title):
    """Returns explicit [[...]] links found inside the note's content."""
    import re
    content = get_note_by_title(title)
    if not content:
        return []
    matches = re.findall(r"\[\[(.*?)\]\]", content)
    return sorted(set([m.strip() for m in matches if m.strip()]))

def get_unlinked_mentions(title):
    """Returns notes that mention the title without an explicit [[...]] link."""
    conn = get_connection()
    mention = f"%{title}%"
    not_link = f"%[[{title}]]%"
    df = conn.execute(
        """
        SELECT DISTINCT title
        FROM notes
        WHERE content ILIKE ? AND content NOT ILIKE ? AND title <> ?
        """,
        (mention, not_link, title)
    ).fetchdf()
    conn.close()
    return df['title'].unique().tolist()

def get_similar_notes(title, limit=5):
    """Returns notes with the most similar embeddings to the given title."""
    conn = get_connection()
    query = conn.execute(
        """
        SELECT embedding
        FROM notes
        WHERE title = ?
        LIMIT 1
        """, (title,)
    ).fetchone()
    if not query or not query[0]:
        return []
    embedding = query[0]
    df = conn.execute(
        """
        SELECT title, 1 - array_cosine_similarity(embedding, ?::FLOAT[]) AS distance
        FROM notes
        WHERE title <> ?
        ORDER BY distance
        LIMIT ?
        """,
        (embedding, title, limit)
    ).fetchdf()
    conn.close()
    return df['title'].tolist()

def upsert_metadata(conn, title, tags, frontmatter):
    """Stores or updates the tags and YAML frontmatter for a note."""
    current_time = datetime.now()
    conn.execute(
        """
        INSERT INTO note_metadata (title, tags, frontmatter, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (title) DO UPDATE SET
            tags = excluded.tags,
            frontmatter = excluded.frontmatter,
            updated_at = excluded.updated_at
        """,
        (title, tags, frontmatter, current_time)
    )

def save_metadata(title, tags, frontmatter):
    """Convenience wrapper to update metadata using a new connection."""
    conn = get_connection()
    upsert_metadata(conn, title, tags, frontmatter)
    conn.close()

def get_metadata(title):
    """Returns tags and frontmatter for a note."""
    conn = get_connection()
    result = conn.execute(
        "SELECT tags, frontmatter FROM note_metadata WHERE title = ?", (title,)
    ).fetchone()
    conn.close()
    if result:
        return {"tags": result[0] or "", "frontmatter": result[1] or "{}", "title": title}
    return {"tags": "", "frontmatter": "{}", "title": title}

def get_all_tags():
    """Returns a sorted list of unique tags."""
    conn = get_connection()
    results = conn.execute("SELECT tags FROM note_metadata").fetchall()
    conn.close()
    tags = set()
    for row in results:
        if row[0]:
            tags.update([t.strip() for t in row[0].split(",") if t.strip()])
    return sorted(tags)

def get_titles_by_tag(tag):
    """Returns titles that contain a given tag."""
    conn = get_connection()
    df = conn.execute(
        "SELECT title FROM note_metadata WHERE tags ILIKE ?",
        (f"%{tag}%",)
    ).fetchdf()
    conn.close()
    return df['title'].unique().tolist()

def update_note_content(title, new_content):
    """Replaces the chunks of a note with newly edited content."""
    if not new_content.strip():
        return
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(new_content)
    if not chunks:
        return
    
    embeddings = get_embeddings(chunks)
    current_time = datetime.now()
    
    conn = get_connection()
    # Remove old chunks for this note
    conn.execute("DELETE FROM notes WHERE title = ?", (title,))
    
    # Insert the regenerated chunks
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{title}_{i}"))
        conn.execute(
            """
            INSERT INTO notes (id, title, content, chunk_index, embedding, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (chunk_id, title, chunk, i, embedding, current_time)
        )
    conn.close()

def delete_note(note_id):
    """Deletes a note."""
    conn = get_connection()
    conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    conn.close()

@st_cache_data(ttl=600)
def advanced_search(query, limit=20):
    """Advanced search with scope operators: tag:, title:, content:, regex:.
    Also supports AND, OR and quoted phrases.
    """
    import re
    conn = get_connection()
    q = query.strip()
    
    sql = """
        SELECT DISTINCT n.id, n.title, n.content, n.updated_at
        FROM notes n
        LEFT JOIN note_metadata m ON n.title = m.title
        WHERE
    """
    params = []
    conditions = []
    
    if q.lower().startswith("regex:"):
        pattern = q[6:].strip()
        conditions.append("n.content ~ ?")
        params.append(pattern)
    else:
        # Tokenize keeping quoted phrases
        tokens = re.findall(r'"[^"]*"|\S+', q)
        tokens = [t.strip('"') for t in tokens]
        
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t.upper() in ("AND", "OR"):
                i += 1
                continue
            if t.upper() == "NOT":
                if i + 1 < len(tokens):
                    nxt = tokens[i + 1]
                    if nxt.lower().startswith("tag:"):
                        conditions.append("(m.tags IS NULL OR m.tags NOT ILIKE ?)")
                        params.append(f"%{nxt[4:].strip()}%")
                    elif nxt.lower().startswith("title:"):
                        conditions.append("n.title NOT ILIKE ?")
                        params.append(f"%{nxt[6:].strip()}%")
                    elif nxt.lower().startswith("content:"):
                        conditions.append("n.content NOT ILIKE ?")
                        params.append(f"%{nxt[8:].strip()}%")
                    else:
                        conditions.append("(n.title NOT ILIKE ? AND n.content NOT ILIKE ?)")
                        params.append(f"%{nxt}%")
                        params.append(f"%{nxt}%")
                    i += 2
                    continue
            elif t.lower().startswith("tag:"):
                conditions.append("m.tags ILIKE ?")
                params.append(f"%{t[4:].strip()}%")
            elif t.lower().startswith("title:"):
                conditions.append("n.title ILIKE ?")
                params.append(f"%{t[6:].strip()}%")
            elif t.lower().startswith("content:"):
                conditions.append("n.content ILIKE ?")
                params.append(f"%{t[8:].strip()}%")
            else:
                conditions.append("(n.title ILIKE ? OR n.content ILIKE ?)")
                params.append(f"%{t}%")
                params.append(f"%{t}%")
            i += 1
    
    if conditions:
        sql += " AND ".join(conditions)
    else:
        sql += "1=1"
    sql += " ORDER BY n.updated_at DESC LIMIT ?"
    params.append(limit)
    
    df = conn.execute(sql, params).fetchdf()
    conn.close()
    return df

@st_cache_data(ttl=600) # Cache keyword search results for 10 minutes
def keyword_search(query, limit=10):
    """Searches notes by keyword in title or content."""
    conn = get_connection()
    q = f"%{query}%"
    sql = """
        SELECT id, title, content, updated_at
        FROM notes
        WHERE title ILIKE ? OR content ILIKE ?
        ORDER BY updated_at DESC
        LIMIT ?
    """
    df = conn.execute(sql, (q, q, limit)).fetchdf()
    conn.close()
    return df

@st_cache_data(ttl=600) # Cache search results for 10 minutes
def search_notes(query, limit=10):
    """Searches notes by semantic similarity."""
    conn = get_connection()
    query_embedding = get_embedding(query)
    
    # DuckDB provides list_cosine_similarity for arrays
    sql = """
        SELECT id, title, content, updated_at,
               list_cosine_similarity(embedding, ?::FLOAT[]) as similarity
        FROM notes
        WHERE embedding IS NOT NULL
        ORDER BY similarity DESC
        LIMIT ?
    """
    
    df = conn.execute(sql, (query_embedding, limit)).fetchdf()
    conn.close()
    return df

def save_summary(title, summary):
    """Saves or updates an LLM-generated summary for a note."""
    conn = get_connection()
    current_time = datetime.now()
    conn.execute(
        """
        INSERT INTO note_summaries (title, summary, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT (title) DO UPDATE SET
            summary = excluded.summary,
            updated_at = excluded.updated_at
        """,
        (title, summary, current_time)
    )
    conn.close()

def get_summary(title):
    """Returns the saved summary for a note, or empty string."""
    conn = get_connection()
    result = conn.execute(
        "SELECT summary FROM note_summaries WHERE title = ?", (title,)
    ).fetchone()
    conn.close()
    return result[0] if result else ""

def list_chat_sessions():
    """Returns all saved chat sessions."""
    conn = get_connection()
    df = conn.execute(
        "SELECT session_id, name, updated_at FROM chat_sessions ORDER BY updated_at DESC"
    ).fetchdf()
    conn.close()
    return df

def load_chat_session(session_id):
    """Loads a chat session as a list of dict messages."""
    import json
    conn = get_connection()
    result = conn.execute(
        "SELECT messages FROM chat_sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    conn.close()
    if not result or not result[0]:
        return []
    try:
        return json.loads(result[0])
    except Exception:
        return []

def save_chat_session(session_id, name, messages):
    """Saves a chat session."""
    import json
    conn = get_connection()
    current_time = datetime.now()
    conn.execute(
        """
        INSERT INTO chat_sessions (session_id, name, messages, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (session_id) DO UPDATE SET
            name = excluded.name,
            messages = excluded.messages,
            updated_at = excluded.updated_at
        """,
        (session_id, name, json.dumps(messages), current_time)
    )
    conn.close()

def delete_chat_session(session_id):
    conn = get_connection()
    conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
    conn.close()

def get_tasks(note_title):
    """Returns tasks for a given note."""
    conn = get_connection()
    df = conn.execute(
        "SELECT id, task_text, completed FROM note_tasks WHERE note_title = ? ORDER BY created_at",
        (note_title,)
    ).fetchdf()
    conn.close()
    return df

def add_task(note_title, task_text, task_id=None):
    """Adds a task to a note."""
    if task_id is None:
        task_id = str(uuid.uuid4())
    conn = get_connection()
    conn.execute(
        "INSERT INTO note_tasks (id, note_title, task_text) VALUES (?, ?, ?)",
        (task_id, note_title, task_text)
    )
    conn.close()
    return task_id

def update_task_status(task_id, completed):
    conn = get_connection()
    conn.execute(
        "UPDATE note_tasks SET completed = ? WHERE id = ?",
        (bool(completed), task_id)
    )
    conn.close()

def delete_task(task_id):
    conn = get_connection()
    conn.execute("DELETE FROM note_tasks WHERE id = ?", (task_id,))
    conn.close()

def get_all_tasks():
    """Returns all tasks with note title."""
    conn = get_connection()
    df = conn.execute(
        "SELECT id, note_title, task_text, completed, created_at FROM note_tasks ORDER BY created_at DESC"
    ).fetchdf()
    conn.close()
    return df

def get_kpi_stats():
    """Returns basic KPIs from the database."""
    conn = get_connection()
    total_chunks = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    total_titles = conn.execute("SELECT COUNT(DISTINCT title) FROM notes").fetchone()[0]
    total_tags = conn.execute("SELECT COUNT(DISTINCT tags) FROM note_metadata").fetchone()[0]
    total_tasks = conn.execute("SELECT COUNT(*) FROM note_tasks").fetchone()[0]
    total_sessions = conn.execute("SELECT COUNT(*) FROM chat_sessions").fetchone()[0]
    recent = conn.execute(
        "SELECT title, MAX(updated_at) as updated_at FROM notes GROUP BY title ORDER BY updated_at DESC LIMIT 5"
    ).fetchdf()
    conn.close()
    return {
        "total_chunks": total_chunks,
        "total_titles": total_titles,
        "total_tags": total_tags,
        "total_tasks": total_tasks,
        "total_sessions": total_sessions,
        "recent": recent
    }

def compare_notes(title_a, title_b):
    """Compares two notes and returns lengths, shared words and embedding similarity."""
    text_a = get_note_by_title(title_a)
    text_b = get_note_by_title(title_b)
    if not text_a or not text_b:
        return None
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    shared = words_a & words_b
    emb_a = get_embedding(text_a[:1000])
    emb_b = get_embedding(text_b[:1000])
    import numpy as np
    similarity = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))
    return {
        "length_a": len(text_a),
        "length_b": len(text_b),
        "words_a": len(words_a),
        "words_b": len(words_b),
        "shared_words": len(shared),
        "shared_sample": list(shared)[:20],
        "similarity": similarity
    }

def extract_entities(title):
    """Extracts dates, laws, rulings and capitalized phrases from a note."""
    import re
    text = get_note_by_title(title)
    if not text:
        return {}
    results = {
        "fechas": sorted(set(re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b", text))),
        "leyes": sorted(set(re.findall(r"(?i)\b(ley|decreto)\s+\d+(?:\s*de\s*\d{4})?", text))),
        "sentencias": sorted(set(re.findall(r"(?i)\b[tsc]-\d+-\d+\b", text))),
        "articulos": sorted(set(re.findall(r"(?i)\bart[ií]culo\s+\d+\b", text))),
        "entidades": sorted(set(re.findall(r"\b[A-Z][A-Z0-9]+(?:\s+[A-Z][A-Z0-9]+){1,4}\b", text)))[:30]
    }
    return results

def build_timeline():
    """Returns a timeline of extracted dates across all notes."""
    import re
    conn = get_connection()
    rows = conn.execute("SELECT DISTINCT title, content FROM notes").fetchall()
    conn.close()
    events = []
    for title, content in rows:
        for match in re.findall(r"\b(\d{1,2}[/-]\d{1,2}[/-](\d{2,4}))\b|\b(\d{4})\b", content):
            date = match[0] if match[0] else match[2]
            if date:
                events.append({"title": title, "date": date, "snippet": content[:120].replace("\n", " ")})
    return sorted(events, key=lambda x: x["date"])

def scan_for_changes(folder_path):
    """Compares files on disk with indexed titles and reports new/missing files."""
    if not os.path.isdir(folder_path):
        return {"error": "Carpeta no encontrada"}
    conn = get_connection()
    indexed = set(conn.execute("SELECT DISTINCT title FROM notes").fetchall())
    conn.close()
    indexed = {t[0] for t in indexed}
    on_disk = set()
    for root, _, files in os.walk(folder_path):
        for f in files:
            on_disk.add(f)
    new_files = on_disk - indexed
    missing = indexed - on_disk
    return {
        "new_files": sorted(new_files),
        "missing_files": sorted(missing),
        "total_indexed": len(indexed),
        "total_on_disk": len(on_disk)
    }

def pin_note(title):
    """Marks a note as pinned."""
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO note_pins (title, pinned, created_at)
        VALUES (?, TRUE, CURRENT_TIMESTAMP)
        ON CONFLICT (title) DO UPDATE SET pinned = TRUE
        """,
        (title,)
    )
    conn.close()

def unpin_note(title):
    """Removes a note from pinned."""
    conn = get_connection()
    conn.execute("DELETE FROM note_pins WHERE title = ?", (title,))
    conn.close()

def is_pinned(title):
    """Returns True if the note is pinned."""
    conn = get_connection()
    result = conn.execute(
        "SELECT pinned FROM note_pins WHERE title = ?", (title,)
    ).fetchone()
    conn.close()
    return bool(result and result[0])

def get_pinned_notes():
    """Returns the list of pinned note titles."""
    conn = get_connection()
    df = conn.execute(
        "SELECT title FROM note_pins WHERE pinned = TRUE ORDER BY created_at DESC"
    ).fetchdf()
    conn.close()
    return df['title'].tolist()

def get_note_embeddings():
    """Returns a dict of title -> embedding for all notes."""
    conn = get_connection()
    rows = conn.execute("SELECT DISTINCT title, embedding FROM notes WHERE embedding IS NOT NULL").fetchall()
    conn.close()
    embeddings = {}
    for title, emb in rows:
        if emb is not None:
            embeddings[title] = emb
    return embeddings

def cluster_notes(similarity_threshold=0.7):
    """Groups notes by embedding cosine similarity into topic clusters."""
    import numpy as np
    embeddings = get_note_embeddings()
    titles = list(embeddings.keys())
    if not titles:
        return []
    vectors = np.array([embeddings[t] for t in titles])
    # Normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-10)
    clusters = []
    used = set()
    for i, title in enumerate(titles):
        if i in used:
            continue
        sims = vectors @ vectors[i]
        cluster = [j for j, s in enumerate(sims) if s >= similarity_threshold and j not in used]
        for j in cluster:
            used.add(j)
        if cluster:
            clusters.append({"label": title, "notes": [titles[j] for j in cluster]})
    return clusters
