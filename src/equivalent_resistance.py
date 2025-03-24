import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory for images if it doesn't exist
image_dir = os.path.join('docs', '1 Physics', '4 Electromagnetism', 'images')
os.makedirs(image_dir, exist_ok=True)

def draw_circuit_graph(G, pos=None, title="Circuit Graph", save_path=None):
    """
    Draw a circuit graph with resistor values as edge labels.
    
    Args:
        G: NetworkX graph representing the circuit
        pos: Dictionary of node positions
        title: Title of the plot
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    if pos is None:
        pos = nx.spring_layout(G, seed=42)  # For consistent layout
    
    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, font_size=12, font_weight='bold')
    
    # Draw edge labels (resistor values)
    edge_labels = {(u, v): f"{d['resistance']:.2f} ohm" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    plt.title(title, fontsize=14)
    plt.axis('off')
    
    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def identify_series_nodes(G):
    """
    Identify nodes that are in series in the graph.
    A node is in series if it has exactly two connections.
    
    Args:
        G: NetworkX graph representing the circuit
        
    Returns:
        List of nodes that are in series (excluding terminals)
    """
    series_nodes = [node for node in G.nodes() if G.degree(node) == 2]
    return series_nodes

def reduce_series(G, node):
    """
    Reduce a series connection at the specified node.
    
    Args:
        G: NetworkX graph representing the circuit
        node: Node to be eliminated (must have exactly two connections)
        
    Returns:
        Modified graph with the series connection reduced
    """
    # Get the two neighbors of the node
    neighbors = list(G.neighbors(node))
    if len(neighbors) != 2:
        raise ValueError(f"Node {node} does not have exactly two connections")
    
    n1, n2 = neighbors
    
    # Get the resistances of the two edges
    r1 = G[n1][node]['resistance']
    r2 = G[node][n2]['resistance']
    
    # Calculate the equivalent resistance
    r_eq = r1 + r2
    
    # Remove the node and its edges
    G.remove_node(node)
    
    # Add a new edge between the neighbors with the equivalent resistance
    G.add_edge(n1, n2, resistance=r_eq)
    
    return G

def identify_parallel_edges(G):
    """
    Identify pairs of nodes that have multiple edges between them (parallel resistors).
    
    Args:
        G: NetworkX graph representing the circuit
        
    Returns:
        List of node pairs that have parallel connections
    """
    # Convert to MultiGraph to find parallel edges
    MG = nx.MultiGraph(G)
    
    parallel_pairs = []
    for u, v, data in MG.edges(data=True):
        if MG.number_of_edges(u, v) > 1:
            if (u, v) not in parallel_pairs and (v, u) not in parallel_pairs:
                parallel_pairs.append((u, v))
    
    return parallel_pairs

def reduce_parallel(G, node_pair):
    """
    Reduce parallel connections between a pair of nodes.
    
    Args:
        G: NetworkX graph representing the circuit
        node_pair: Tuple of nodes that have parallel connections
        
    Returns:
        Modified graph with the parallel connections reduced
    """
    u, v = node_pair
    
    # Get all edges between the nodes
    edges = []
    for n1, n2, data in G.edges(data=True):
        if (n1 == u and n2 == v) or (n1 == v and n2 == u):
            edges.append(data['resistance'])
    
    # Calculate the equivalent resistance (1/R_eq = 1/R1 + 1/R2 + ...)
    r_eq = 1.0 / sum(1.0 / r for r in edges)
    
    # Remove all edges between the nodes
    while G.has_edge(u, v):
        G.remove_edge(u, v)
    
    # Add a new edge with the equivalent resistance
    G.add_edge(u, v, resistance=r_eq)
    
    return G

def delta_to_wye_transformation(G, nodes):
    """
    Apply the Delta to Wye (Delta to Y) transformation for three nodes forming a triangle.
    
    Args:
        G: NetworkX graph representing the circuit
        nodes: Three nodes forming a delta (triangle)
        
    Returns:
        Modified graph with the delta transformed to wye
    """
    if len(nodes) != 3:
        raise ValueError("Delta to Wye transformation requires exactly three nodes")
    
    a, b, c = nodes
    
    # Get the resistances of the three edges
    if not (G.has_edge(a, b) and G.has_edge(b, c) and G.has_edge(c, a)):
        raise ValueError("The three nodes must form a complete triangle")
    
    r_ab = G[a][b]['resistance']
    r_bc = G[b][c]['resistance']
    r_ca = G[c][a]['resistance']
    
    # Calculate the equivalent resistances for the Y configuration
    denominator = r_ab + r_bc + r_ca
    r_a = (r_ab * r_ca) / denominator
    r_b = (r_ab * r_bc) / denominator
    r_c = (r_bc * r_ca) / denominator
    
    # Create a new central node
    new_node = max(G.nodes()) + 1
    
    # Remove the original delta edges
    G.remove_edge(a, b)
    G.remove_edge(b, c)
    G.remove_edge(c, a)
    
    # Add the new Y edges
    G.add_edge(a, new_node, resistance=r_a)
    G.add_edge(b, new_node, resistance=r_b)
    G.add_edge(c, new_node, resistance=r_c)
    
    return G

def find_delta_configurations(G):
    """
    Find all delta (triangle) configurations in the graph.
    
    Args:
        G: NetworkX graph representing the circuit
        
    Returns:
        List of node triplets forming delta configurations
    """
    delta_configs = []
    
    # Check all possible triplets of nodes
    for a in G.nodes():
        for b in G.neighbors(a):
            for c in G.neighbors(a):
                if b != c and G.has_edge(b, c):
                    # Found a triangle (a, b, c)
                    # Sort the nodes to avoid duplicates
                    triangle = tuple(sorted([a, b, c]))
                    if triangle not in delta_configs:
                        delta_configs.append(triangle)
    
    return delta_configs

def calculate_equivalent_resistance(G, source, target):
    """
    Calculate the equivalent resistance between two nodes in a circuit.
    
    Args:
        G: NetworkX graph representing the circuit
        source: Source node
        target: Target node
        
    Returns:
        Equivalent resistance between source and target
    """
    # Make a copy of the graph to avoid modifying the original
    H = G.copy()
    
    # Keep track of the reduction steps for visualization
    reduction_steps = []
    reduction_steps.append((H.copy(), "Initial Circuit"))
    
    # Continue reducing the graph until only the source and target nodes remain
    iteration = 0
    max_iterations = 100  # Prevent infinite loops
    
    while len(H.nodes()) > 2 and iteration < max_iterations:
        iteration += 1
        
        # Try to reduce series connections
        series_nodes = identify_series_nodes(H)
        
        # Filter out source and target nodes
        series_nodes = [node for node in series_nodes if node != source and node != target]
        
        if series_nodes:
            # Reduce a series connection
            node = series_nodes[0]
            H = reduce_series(H, node)
            reduction_steps.append((H.copy(), f"After Series Reduction at Node {node}"))
            continue
        
        # Try to reduce parallel connections
        parallel_pairs = identify_parallel_edges(H)
        if parallel_pairs:
            # Reduce a parallel connection
            pair = parallel_pairs[0]
            H = reduce_parallel(H, pair)
            reduction_steps.append((H.copy(), f"After Parallel Reduction between Nodes {pair}"))
            continue
        
        # Try to apply delta-wye transformation
        delta_configs = find_delta_configurations(H)
        if delta_configs:
            # Filter out configurations that include source or target
            delta_configs = [config for config in delta_configs 
                            if source not in config and target not in config]
            
            if delta_configs:
                # Apply delta-wye transformation
                config = delta_configs[0]
                try:
                    H = delta_to_wye_transformation(H, config)
                    reduction_steps.append((H.copy(), f"After Delta-Wye Transformation at Nodes {config}"))
                    continue
                except Exception as e:
                    print(f"Error in delta-wye transformation: {e}")
        
        # If no reductions are possible, break the loop
        break
    
    # Check if the reduction was successful
    if len(H.nodes()) == 2 and H.has_edge(source, target):
        equivalent_resistance = H[source][target]['resistance']
    else:
        # For more complex circuits, we might need to use other methods
        raise ValueError("Could not reduce the circuit completely. Try using delta-wye transformations or other methods.")
    
    return equivalent_resistance, reduction_steps

def create_example_circuits():
    """
    Create example circuits for testing the algorithm.
    
    Returns:
        List of (graph, source, target, description) tuples
    """
    examples = []
    
    # Example 1: Simple series circuit
    G1 = nx.Graph()
    G1.add_edge(0, 1, resistance=10.0)
    G1.add_edge(1, 2, resistance=20.0)
    G1.add_edge(2, 3, resistance=30.0)
    examples.append((G1, 0, 3, "Simple Series Circuit"))
    
    # Example 2: Simple parallel circuit
    G2 = nx.Graph()
    G2.add_edge(0, 1, resistance=10.0)
    G2.add_edge(0, 1, resistance=20.0)  # Parallel edge
    examples.append((G2, 0, 1, "Simple Parallel Circuit"))
    
    # Example 3: Mixed series-parallel circuit
    G3 = nx.Graph()
    G3.add_edge(0, 1, resistance=10.0)
    G3.add_edge(1, 2, resistance=20.0)
    G3.add_edge(1, 3, resistance=30.0)
    G3.add_edge(2, 4, resistance=40.0)
    G3.add_edge(3, 4, resistance=50.0)
    G3.add_edge(4, 5, resistance=60.0)
    examples.append((G3, 0, 5, "Mixed Series-Parallel Circuit"))
    
    return examples

def analyze_example_circuits():
    """
    Analyze example circuits and save the results.
    """
    examples = create_example_circuits()
    
    results = []
    
    for i, (G, source, target, description) in enumerate(examples):
        print(f"\nAnalyzing {description}...")
        
        # Draw the initial circuit
        pos = nx.spring_layout(G, seed=42)  # For consistent layout
        draw_circuit_graph(G, pos, 
                          title=f"Example {i+1}: {description}",
                          save_path=os.path.join(image_dir, f"example_{i+1}_initial.png"))
        
        try:
            # Calculate the equivalent resistance
            equivalent_resistance, reduction_steps = calculate_equivalent_resistance(G, source, target)
            
            print(f"Equivalent resistance: {equivalent_resistance:.2f} ohm")
            
            # Draw the reduction steps
            for j, (H, step_description) in enumerate(reduction_steps):
                draw_circuit_graph(H, pos, 
                                  title=f"Example {i+1}: {step_description}",
                                  save_path=os.path.join(image_dir, f"example_{i+1}_step_{j}.png"))
            
            results.append({
                "example": i+1,
                "description": description,
                "equivalent_resistance": equivalent_resistance,
                "num_steps": len(reduction_steps) - 1  # Subtract 1 for the initial state
            })
            
        except Exception as e:
            print(f"Error: {e}")
    
    return results

if __name__ == "__main__":
    # Analyze example circuits
    results = analyze_example_circuits()
    
    # Print a summary of the results
    print("\nSummary of Results:")
    print("-" * 80)
    print(f"{'Example':<10} {'Description':<30} {'Equivalent Resistance':<25} {'Steps':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['example']:<10} {result['description']:<30} {result['equivalent_resistance']:.2f} ohm{' ':<15} {result['num_steps']:<10}")
    
    print("\nAll analyses completed.")
    print(f"Images saved to {image_dir}")
