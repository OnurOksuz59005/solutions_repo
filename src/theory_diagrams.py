import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import os

# Create directory for images if it doesn't exist
image_dir = os.path.join('docs', '1 Physics', '4 Electromagnetism', 'images')
os.makedirs(image_dir, exist_ok=True)

def draw_resistor(ax, start, end, value=None, orientation='horizontal', offset=0.4, color='k'):
    """
    Draw a resistor symbol between start and end points.
    
    Args:
        ax: Matplotlib axis
        start: Starting point (x, y)
        end: Ending point (x, y)
        value: Resistance value to display
        orientation: 'horizontal' or 'vertical'
        offset: Text offset for the label
        color: Color of the resistor
    """
    # Calculate direction and length
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = np.sqrt(dx**2 + dy**2)
    
    # Normalize direction
    if length > 0:
        dx /= length
        dy /= length
    
    # Calculate perpendicular direction for zigzag
    perpx = -dy
    perpy = dx
    
    # Calculate zigzag points
    num_segments = 6
    zigzag_length = length * 0.6
    zigzag_height = 0.2
    
    # Calculate start and end of zigzag
    zigzag_start = (start[0] + dx * (length - zigzag_length) / 2, 
                   start[1] + dy * (length - zigzag_length) / 2)
    
    # Draw the lines before and after zigzag
    ax.plot([start[0], zigzag_start[0]], [start[1], zigzag_start[1]], color=color, lw=2)
    
    zigzag_end = (start[0] + dx * (length + zigzag_length) / 2, 
                 start[1] + dy * (length + zigzag_length) / 2)
    
    ax.plot([zigzag_end[0], end[0]], [zigzag_end[1], end[1]], color=color, lw=2)
    
    # Draw the zigzag
    zigzag_points_x = []
    zigzag_points_y = []
    
    for i in range(num_segments + 1):
        t = i / num_segments
        point_x = zigzag_start[0] + dx * zigzag_length * t
        point_y = zigzag_start[1] + dy * zigzag_length * t
        
        # Add zigzag effect
        if i % 2 == 1:
            point_x += perpx * zigzag_height
            point_y += perpy * zigzag_height
        else:
            point_x -= perpx * zigzag_height
            point_y -= perpy * zigzag_height
        
        zigzag_points_x.append(point_x)
        zigzag_points_y.append(point_y)
    
    ax.plot(zigzag_points_x, zigzag_points_y, color=color, lw=2)
    
    # Add label if value is provided
    if value is not None:
        # Position for the label - always on top of the resistor
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        
        if orientation == 'horizontal':
            label_x = mid_x
            label_y = mid_y + offset
            ha = 'center'
            va = 'bottom'
        else:  # vertical
            label_x = mid_x + offset
            label_y = mid_y
            ha = 'left'
            va = 'center'
        
        # Ensure the text is spaced away from the resistor
        ax.text(label_x, label_y, value, ha=ha, va=va, fontsize=16, color='#FF5733', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))

def draw_node(ax, pos, node_id=None, color='k', size=6):
    """
    Draw a node (connection point) at the specified position.
    
    Args:
        ax: Matplotlib axis
        pos: (x, y) position
        node_id: Optional node identifier to display
        color: Color of the node
        size: Size of the node
    """
    ax.plot(pos[0], pos[1], 'o', color=color, markersize=size)
    
    if node_id is not None:
        ax.text(pos[0], pos[1] - 0.2, str(node_id), ha='center', va='center', fontsize=16, fontweight='bold')

def draw_wire(ax, points, color='k'):
    """
    Draw a wire connecting multiple points.
    
    Args:
        ax: Matplotlib axis
        points: List of (x, y) points to connect
        color: Color of the wire
    """
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    
    ax.plot(x_vals, y_vals, color=color, lw=2)

def draw_graph_representation():
    """
    Draw a circuit and its graph representation side by side.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Draw the circuit on the left
    # Define node positions for the circuit
    circuit_nodes = {
        'A': (1, 3),
        'B': (3, 3),
        'C': (3, 1),
        'D': (1, 1)
    }
    
    # Draw nodes
    for node_id, pos in circuit_nodes.items():
        draw_node(ax1, pos, node_id)
    
    # Draw resistors
    draw_resistor(ax1, circuit_nodes['A'], circuit_nodes['B'], "R₁ = 10Ω")
    draw_resistor(ax1, circuit_nodes['B'], circuit_nodes['C'], "R₂ = 20Ω")
    draw_resistor(ax1, circuit_nodes['C'], circuit_nodes['D'], "R₃ = 15Ω")
    draw_resistor(ax1, circuit_nodes['D'], circuit_nodes['A'], "R₄ = 30Ω")
    
    # Set axis properties for circuit
    ax1.set_xlim(0, 4)
    ax1.set_ylim(0, 4)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title("Circuit Diagram", fontsize=18, fontweight='bold')
    
    # Draw the graph representation on the right
    G = nx.Graph()
    G.add_nodes_from(['A', 'B', 'C', 'D'])
    G.add_edge('A', 'B', weight=1)
    G.add_edge('B', 'C', weight=1)
    G.add_edge('C', 'D', weight=1)
    G.add_edge('D', 'A', weight=1)
    
    # Position nodes in a similar layout to the circuit
    pos = {
        'A': (0, 1),
        'B': (1, 1),
        'C': (1, 0),
        'D': (0, 0)
    }
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=600, ax=ax2)
    nx.draw_networkx_labels(G, pos, font_size=16, font_weight='bold', ax=ax2)
    
    # Draw edges with labels
    nx.draw_networkx_edges(G, pos, width=2, ax=ax2)
    edge_labels = {('A', 'B'): 'R₁ = 10Ω', ('B', 'C'): 'R₂ = 20Ω', ('C', 'D'): 'R₃ = 15Ω', ('D', 'A'): 'R₄ = 30Ω'}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=14, font_weight='bold', ax=ax2, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))
    
    # Set axis properties for graph
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(-0.5, 1.5)
    ax2.axis('off')
    ax2.set_title("Graph Representation", fontsize=18, fontweight='bold')
    
    # Add overall title
    plt.suptitle("Circuit to Graph Transformation", fontsize=20, fontweight='bold')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(image_dir, "graph_representation.png"), dpi=300, bbox_inches='tight')
    plt.close()

def draw_series_reduction():
    """
    Draw the series reduction process.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Draw the original circuit on the left
    # Define node positions
    nodes1 = {
        'A': (1, 2),
        'B': (3, 2),
        'C': (5, 2)
    }
    
    # Draw nodes
    for node_id, pos in nodes1.items():
        draw_node(ax1, pos, node_id)
    
    # Draw resistors
    draw_resistor(ax1, nodes1['A'], nodes1['B'], "R₁ = 10Ω")
    draw_resistor(ax1, nodes1['B'], nodes1['C'], "R₂ = 20Ω")
    
    # Set axis properties
    ax1.set_xlim(0, 6)
    ax1.set_ylim(0, 3)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title("Before Series Reduction", fontsize=18, fontweight='bold')
    
    # Draw the reduced circuit on the right
    # Define node positions
    nodes2 = {
        'A': (1, 2),
        'C': (5, 2)
    }
    
    # Draw nodes
    for node_id, pos in nodes2.items():
        draw_node(ax2, pos, node_id)
    
    # Draw the equivalent resistor
    draw_resistor(ax2, nodes2['A'], nodes2['C'], "Rₑq = 30Ω")
    
    # Set axis properties
    ax2.set_xlim(0, 6)
    ax2.set_ylim(0, 3)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title("After Series Reduction", fontsize=18, fontweight='bold')
    
    # Add overall title
    plt.suptitle("Series Reduction", fontsize=20, fontweight='bold')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(image_dir, "series_reduction.png"), dpi=300, bbox_inches='tight')
    plt.close()

def draw_parallel_reduction():
    """
    Draw the parallel reduction process.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Draw the original circuit on the left
    # Define node positions
    nodes1 = {
        'A': (1, 3),
        'B': (5, 3),
        'C': (5, 1),
        'D': (1, 1)
    }
    
    # Draw nodes
    for node_id, pos in nodes1.items():
        draw_node(ax1, pos, node_id)
    
    # Draw resistors
    draw_resistor(ax1, nodes1['A'], nodes1['B'], "R₁ = 30Ω")
    draw_resistor(ax1, nodes1['D'], nodes1['C'], "R₂ = 60Ω")
    
    # Draw wires
    draw_wire(ax1, [nodes1['B'], nodes1['C']])
    draw_wire(ax1, [nodes1['D'], nodes1['A']])
    
    # Set axis properties
    ax1.set_xlim(0, 6)
    ax1.set_ylim(0, 4)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title("Before Parallel Reduction", fontsize=18, fontweight='bold')
    
    # Draw the reduced circuit on the right
    # Define node positions
    nodes2 = {
        'A': (1, 2),
        'B': (5, 2)
    }
    
    # Draw nodes
    for node_id, pos in nodes2.items():
        draw_node(ax2, pos, node_id)
    
    # Draw the equivalent resistor
    draw_resistor(ax2, nodes2['A'], nodes2['B'], "Rₑq = 20Ω")
    
    # Set axis properties
    ax2.set_xlim(0, 6)
    ax2.set_ylim(0, 4)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title("After Parallel Reduction", fontsize=18, fontweight='bold')
    
    # Add overall title
    plt.suptitle("Parallel Reduction", fontsize=20, fontweight='bold')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(image_dir, "parallel_reduction.png"), dpi=300, bbox_inches='tight')
    plt.close()

def draw_delta_wye_transformation():
    """
    Draw the Delta-Wye transformation process.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Draw the Delta configuration on the left
    # Define node positions
    delta_nodes = {
        'A': (2, 3.5),
        'B': (1, 1),
        'C': (3, 1)
    }
    
    # Draw nodes
    for node_id, pos in delta_nodes.items():
        draw_node(ax1, pos, node_id)
    
    # Draw resistors
    draw_resistor(ax1, delta_nodes['A'], delta_nodes['B'], "Rₐb = 10Ω")
    draw_resistor(ax1, delta_nodes['B'], delta_nodes['C'], "Rbₐc = 20Ω")
    draw_resistor(ax1, delta_nodes['C'], delta_nodes['A'], "Rₐc = 30Ω")
    
    # Set axis properties
    ax1.set_xlim(0, 4)
    ax1.set_ylim(0, 4)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title("Delta (Δ) Configuration", fontsize=18, fontweight='bold')
    
    # Draw the Wye configuration on the right
    # Define node positions
    wye_nodes = {
        'A': (2, 3.5),
        'B': (1, 1),
        'C': (3, 1),
        'O': (2, 2)  # Center node
    }
    
    # Draw nodes
    for node_id, pos in wye_nodes.items():
        if node_id == 'O':
            draw_node(ax2, pos, node_id, color='red', size=8)
        else:
            draw_node(ax2, pos, node_id)
    
    # Draw resistors
    draw_resistor(ax2, wye_nodes['A'], wye_nodes['O'], "Rₐ = 5Ω")
    draw_resistor(ax2, wye_nodes['B'], wye_nodes['O'], "Rb = 3.33Ω")
    draw_resistor(ax2, wye_nodes['C'], wye_nodes['O'], "Rₐ = 10Ω")
    
    # Set axis properties
    ax2.set_xlim(0, 4)
    ax2.set_ylim(0, 4)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title("Wye (Y) Configuration", fontsize=18, fontweight='bold')
    
    # Add overall title
    plt.suptitle("Delta-Wye (Δ-Y) Transformation", fontsize=20, fontweight='bold')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(image_dir, "delta_wye_transformation.png"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Draw all theoretical diagrams
    draw_graph_representation()
    draw_series_reduction()
    draw_parallel_reduction()
    draw_delta_wye_transformation()
    
    print("All theoretical diagrams created.")
    print(f"Images saved to {image_dir}")
