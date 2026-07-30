import networkx as nx
import os
import pickle
from pyvis.network import Network

GRAPH_PATH = 'obsidian_graph.gpickle'

def get_graph():
    """Carga el grafo desde el archivo local o crea uno nuevo si no existe."""
    if os.path.exists(GRAPH_PATH):
        try:
            with open(GRAPH_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return nx.DiGraph()
    return nx.DiGraph()

def save_graph(G):
    """Guarda el grafo en un archivo local."""
    with open(GRAPH_PATH, 'wb') as f:
        pickle.dump(G, f)

def update_graph_edges(source_node, target_nodes):
    """
    Actualiza las aristas del grafo para un nodo fuente.
    Elimina las aristas antiguas de este nodo y añade las nuevas.
    """
    G = get_graph()
    
    # Asegurar que el nodo fuente existe
    if not G.has_node(source_node):
        G.add_node(source_node)
        
    # Eliminar aristas salientes actuales de este nodo
    out_edges = list(G.out_edges(source_node))
    G.remove_edges_from(out_edges)
    
    # Añadir nuevas aristas
    for target in target_nodes:
        if not G.has_node(target):
            G.add_node(target)
        G.add_edge(source_node, target)
        
    save_graph(G)

def get_neighbors(node, degree=1):
    """
    Obtiene los vecinos de un nodo hasta un cierto grado.
    """
    G = get_graph()
    if not G.has_node(node):
        return []
        
    neighbors = set()
    current_level = {node}
    
    for _ in range(degree):
        next_level = set()
        for n in current_level:
            # Añadir vecinos directos (salientes y entrantes)
            next_level.update(G.successors(n))
            next_level.update(G.predecessors(n))
        
        neighbors.update(next_level)
        current_level = next_level
        
    # Remover el nodo original si está en la lista
    if node in neighbors:
        neighbors.remove(node)
        
    return list(neighbors)

def add_nodes_from_titles(titles):
    """Adds nodes from a list of titles without creating edges."""
    G = get_graph()
    for title in titles:
        if not G.has_node(title):
            G.add_node(title)
    save_graph(G)

def rebuild_graph(notes):
    """Rebuilds the graph from notes, detecting [[...]] links between them."""
    import re
    G = get_graph()
    G.clear()
    for note in notes:
        title = note['title']
        content = note['content'] or ""
        G.add_node(title)
        for match in re.findall(r"\[\[(.*?)\]\]", content):
            target = match.strip()
            if target:
                G.add_node(target)
                G.add_edge(title, target)
    save_graph(G)

def generate_graph_html(output_file="graph_view.html"):
    """
    Genera un archivo HTML con el grafo interactivo usando PyVis.
    """
    G = get_graph()
    
    if not G or len(G.nodes) == 0:
        raise ValueError("El grafo está vacío. Ingesta documentos o haz clic en Reconstruir.")
    
    # Crear red de pyvis con recursos embebidos (funciona offline)
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#222222",
        font_color="white",
        directed=True,
        cdn_resources="in_line",
    )
    
    # Configuraciones de física para que se parezca al Graph View de Obsidian
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08, damping=0.4, overlap=0)
    
    # Convertir grafo de networkx a pyvis
    net.from_nx(G)
    
    # Generar HTML y guardar con UTF-8 para evitar errores de codec en Windows
    html = net.generate_html()
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    return output_file

def get_graph_metrics():
    """Returns basic graph metrics: nodes, edges, top connected, clusters."""
    G = get_graph()
    if not G or len(G.nodes) == 0:
        return {}
    degrees = dict(G.degree())
    top_connected = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    clusters = list(nx.weakly_connected_components(G))
    cluster_sizes = [len(c) for c in clusters]
    return {
        "nodes": len(G.nodes),
        "edges": len(G.edges),
        "top_connected": top_connected,
        "clusters": len(clusters),
        "largest_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
    }

def generate_local_graph_html(title, output_file="local_graph.html", degree=1):
    """Generates an HTML graph for the neighborhood of a single note."""
    G = get_graph()
    if not G.has_node(title):
        raise ValueError(f"La nota '{title}' no está en el grafo.")
    
    neighbors = get_neighbors(title, degree)
    nodes = {title} | set(neighbors)
    sub = G.subgraph(nodes).copy()
    
    net = Network(
        height="400px",
        width="100%",
        bgcolor="#222222",
        font_color="white",
        directed=True,
        cdn_resources="in_line",
    )
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08, damping=0.4, overlap=0)
    net.from_nx(sub)
    html = net.generate_html()
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    return output_file

