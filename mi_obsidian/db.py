import duckdb
import os
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer

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

@st_cache_resource
def get_model():
    global model
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
    if close_conn:
        conn.close()

def get_embedding(text):
    """Generates an embedding for the given text."""
    embed_model = get_model()
    return embed_model.encode(text).tolist()

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

@st_cache_data
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

def delete_note(note_id):
    """Deletes a note."""
    conn = get_connection()
    conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    conn.close()

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
