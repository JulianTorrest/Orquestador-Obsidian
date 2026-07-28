from mcp.server.fastmcp import FastMCP
import db
import graph_store
import retriever
import os

# Inicializar FastMCP Server
mcp = FastMCP("Nexus Obsidian Vault")

@mcp.tool()
def search_vault(query: str, top_k: int = 3) -> str:
    """
    Busca notas en la bóveda usando búsqueda híbrida y semántica.
    """
    try:
        results = retriever.hybrid_search_and_rerank(query, top_k=top_k)
        if not results:
            return "No se encontraron resultados en la bóveda."
            
        formatted = []
        for doc in results:
            title = doc.get("title", "Desconocido")
            content = doc.get("content", "")
            formatted.append(f"### Nota: {title}\n{content}")
            
        return "\n\n---\n\n".join(formatted)
    except Exception as e:
        return f"Error en la búsqueda: {str(e)}"

@mcp.tool()
def get_graph_neighbors(note_name: str, degree: int = 1) -> str:
    """
    Obtiene las notas relacionadas a una nota específica mediante los enlaces bidireccionales.
    """
    try:
        neighbors = graph_store.get_neighbors(note_name, degree=degree)
        if not neighbors:
            return f"No se encontraron conexiones para la nota '{note_name}'."
            
        return f"Notas relacionadas a {note_name}:\n" + "\n".join([f"- {n}" for n in neighbors])
    except Exception as e:
        return f"Error consultando el grafo: {str(e)}"

@mcp.tool()
def read_note(note_name: str) -> str:
    """
    Lee el contenido completo de una nota si existe en la base de datos de fragmentos.
    """
    try:
        all_notes = db.get_all_notes_full_content()
        # Filtrar por título
        note_chunks = [n["content"] for n in all_notes if n["title"] == note_name]
        
        if not note_chunks:
            return f"La nota '{note_name}' no existe en la bóveda."
            
        return f"### {note_name}\n\n" + "\n\n".join(note_chunks)
    except Exception as e:
        return f"Error leyendo la nota: {str(e)}"

if __name__ == "__main__":
    # Ejecuta el servidor MCP vía stdio (el estándar para Claude Desktop y Cursor)
    mcp.run()
