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

def generate_graph_html(output_file="graph_view.html"):
    """
    Genera un archivo HTML con el grafo interactivo usando PyVis.
    """
    G = get_graph()
    
    # Crear red de pyvis
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", directed=True)
    
    # Configuraciones de física para que se parezca al Graph View de Obsidian
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08, damping=0.4, overlap=0)
    
    # Convertir grafo de networkx a pyvis
    net.from_nx(G)
    
    # Guardar en archivo
    net.save_graph(output_file)
    return output_file

