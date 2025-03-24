import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Create directory for images if it doesn't exist
image_dir = os.path.join('docs', '1 Physics', '4 Electromagnetism', 'images')
os.makedirs(image_dir, exist_ok=True)

def draw_resistor(ax, start, end, value=None, orientation='horizontal', offset=0):
    """
    Draw a resistor symbol between start and end points.
    
    Args:
        ax: Matplotlib axis
        start: Starting point (x, y)
        end: Ending point (x, y)
        value: Resistance value to display
        orientation: 'horizontal' or 'vertical'
        offset: Text offset for the label
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
    ax.plot([start[0], zigzag_start[0]], [start[1], zigzag_start[1]], 'k-', lw=2)
    
    zigzag_end = (start[0] + dx * (length + zigzag_length) / 2, 
                 start[1] + dy * (length + zigzag_length) / 2)
    
    ax.plot([zigzag_end[0], end[0]], [zigzag_end[1], end[1]], 'k-', lw=2)
    
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
    
    ax.plot(zigzag_points_x, zigzag_points_y, 'k-', lw=2)
    
    # Add label if value is provided
    if value is not None:
        # Position for the label
        if orientation == 'horizontal':
            label_x = (start[0] + end[0]) / 2
            label_y = (start[1] + end[1]) / 2 + offset
            ha = 'center'
            va = 'bottom' if offset > 0 else 'top'
        else:  # vertical
            label_x = (start[0] + end[0]) / 2 + offset
            label_y = (start[1] + end[1]) / 2
            ha = 'left' if offset > 0 else 'right'
            va = 'center'
        
        ax.text(label_x, label_y, f'R = {value} Ω', ha=ha, va=va, fontsize=10, color='#FF5733')

def draw_battery(ax, start, end, voltage=None):
    """
    Draw a battery symbol between start and end points.
    
    Args:
        ax: Matplotlib axis
        start: Starting point (x, y) - negative terminal
        end: Ending point (x, y) - positive terminal
        voltage: Voltage value to display
    """
    # Calculate direction and length
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = np.sqrt(dx**2 + dy**2)
    
    # Normalize direction
    if length > 0:
        dx /= length
        dy /= length
    
    # Calculate perpendicular direction
    perpx = -dy
    perpy = dx
    
    # Draw the line
    ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', lw=2)
    
    # Calculate positions for the battery symbol
    small_gap = length * 0.1
    large_gap = length * 0.2
    
    # Position for the negative terminal (shorter line)
    neg_pos = (start[0] + dx * small_gap, start[1] + dy * small_gap)
    neg_length = 0.1
    
    # Draw negative terminal (shorter line)
    ax.plot([neg_pos[0] - perpx * neg_length, neg_pos[0] + perpx * neg_length],
            [neg_pos[1] - perpy * neg_length, neg_pos[1] + perpy * neg_length], 'k-', lw=2)
    
    # Position for the positive terminal (longer line)
    pos_pos = (start[0] + dx * (small_gap + large_gap), start[1] + dy * (small_gap + large_gap))
    pos_length = 0.2
    
    # Draw positive terminal (longer line)
    ax.plot([pos_pos[0] - perpx * pos_length, pos_pos[0] + perpx * pos_length],
            [pos_pos[1] - perpy * pos_length, pos_pos[1] + perpy * pos_length], 'k-', lw=2)
    
    # Add label if voltage is provided
    if voltage is not None:
        # Position for the label
        label_x = (start[0] + end[0]) / 2 - perpx * 0.3
        label_y = (start[1] + end[1]) / 2 - perpy * 0.3
        
        ax.text(label_x, label_y, f'{voltage} V', ha='center', va='center', fontsize=10, color='#FF5733')

def draw_wire(ax, points):
    """
    Draw a wire connecting multiple points.
    
    Args:
        ax: Matplotlib axis
        points: List of (x, y) points to connect
    """
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    
    ax.plot(x_vals, y_vals, 'k-', lw=2)

def draw_node(ax, pos, node_id=None):
    """
    Draw a node (connection point) at the specified position.
    
    Args:
        ax: Matplotlib axis
        pos: (x, y) position
        node_id: Optional node identifier to display
    """
    ax.plot(pos[0], pos[1], 'ko', markersize=6)
    
    if node_id is not None:
        ax.text(pos[0], pos[1] - 0.2, str(node_id), ha='center', va='center', fontsize=12)

def draw_series_circuit():
    """
    Draw a simple series circuit with three resistors.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define node positions
    nodes = {
        0: (1, 3),
        1: (3, 3),
        2: (5, 3),
        3: (7, 3),
        4: (7, 1),
        5: (1, 1)
    }
    
    # Draw nodes
    for node_id, pos in nodes.items():
        draw_node(ax, pos, node_id)
    
    # Draw resistors
    draw_resistor(ax, nodes[0], nodes[1], "10 kΩ", offset=0.3)
    draw_resistor(ax, nodes[1], nodes[2], "20 kΩ", offset=0.3)
    draw_resistor(ax, nodes[2], nodes[3], "30 kΩ", offset=0.3)
    
    # Draw wires
    draw_wire(ax, [nodes[3], nodes[4]])
    draw_wire(ax, [nodes[4], nodes[5]])
    
    # Draw battery
    draw_battery(ax, nodes[5], nodes[0], "9")
    
    # Set axis properties
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title
    ax.set_title("Example 1: Simple Series Circuit", fontsize=14)
    
    # Save the figure
    plt.savefig(os.path.join(image_dir, "example_1_circuit.png"), dpi=300, bbox_inches='tight')
    plt.close()

def draw_parallel_circuit():
    """
    Draw a simple parallel circuit with two resistors.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define node positions
    nodes = {
        0: (1, 3),
        1: (5, 3),
        2: (5, 1),
        3: (1, 1)
    }
    
    # Draw nodes
    for node_id, pos in nodes.items():
        draw_node(ax, pos, node_id)
    
    # Draw resistors
    draw_resistor(ax, nodes[0], nodes[1], "10 kΩ", offset=0.3)
    draw_resistor(ax, nodes[3], nodes[2], "20 kΩ", offset=0.3)
    
    # Draw wires
    draw_wire(ax, [nodes[1], nodes[2]])
    
    # Draw battery
    draw_battery(ax, nodes[3], nodes[0], "9")
    
    # Set axis properties
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title
    ax.set_title("Example 2: Simple Parallel Circuit", fontsize=14)
    
    # Save the figure
    plt.savefig(os.path.join(image_dir, "example_2_circuit.png"), dpi=300, bbox_inches='tight')
    plt.close()

def draw_mixed_circuit():
    """
    Draw a mixed series-parallel circuit.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define node positions
    nodes = {
        0: (1, 3),
        1: (3, 3),
        2: (5, 4),
        3: (5, 2),
        4: (7, 3),
        5: (9, 3),
        6: (9, 1),
        7: (1, 1)
    }
    
    # Draw nodes
    for node_id, pos in nodes.items():
        draw_node(ax, pos, node_id)
    
    # Draw resistors
    draw_resistor(ax, nodes[0], nodes[1], "10 kΩ", offset=0.3)
    draw_resistor(ax, nodes[1], nodes[2], "20 kΩ", offset=0.3)
    draw_resistor(ax, nodes[1], nodes[3], "30 kΩ", offset=-0.3)
    draw_resistor(ax, nodes[2], nodes[4], "40 kΩ", offset=0.3)
    draw_resistor(ax, nodes[3], nodes[4], "50 kΩ", offset=-0.3)
    draw_resistor(ax, nodes[4], nodes[5], "60 kΩ", offset=0.3)
    
    # Draw wires
    draw_wire(ax, [nodes[5], nodes[6]])
    draw_wire(ax, [nodes[6], nodes[7]])
    
    # Draw battery
    draw_battery(ax, nodes[7], nodes[0], "9")
    
    # Set axis properties
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title
    ax.set_title("Example 3: Mixed Series-Parallel Circuit", fontsize=14)
    
    # Save the figure
    plt.savefig(os.path.join(image_dir, "example_3_circuit.png"), dpi=300, bbox_inches='tight')
    plt.close()

def draw_wheatstone_bridge():
    """
    Draw a Wheatstone bridge circuit.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define node positions
    nodes = {
        0: (1, 3),
        1: (5, 5),
        2: (5, 1),
        3: (9, 3),
        4: (5, 3)  # Center node for the bridge
    }
    
    # Draw nodes
    for node_id, pos in nodes.items():
        draw_node(ax, pos, node_id)
    
    # Draw resistors
    draw_resistor(ax, nodes[0], nodes[1], "10 kΩ", offset=0.3)
    draw_resistor(ax, nodes[0], nodes[2], "20 kΩ", offset=-0.3)
    draw_resistor(ax, nodes[1], nodes[3], "30 kΩ", offset=0.3)
    draw_resistor(ax, nodes[2], nodes[3], "40 kΩ", offset=-0.3)
    draw_resistor(ax, nodes[1], nodes[2], "50 kΩ", orientation='vertical', offset=0.3)
    
    # Draw battery
    draw_battery(ax, nodes[0], nodes[3], "9")
    
    # Set axis properties
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title
    ax.set_title("Example 4: Wheatstone Bridge Circuit", fontsize=14)
    
    # Save the figure
    plt.savefig(os.path.join(image_dir, "example_4_circuit.png"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Draw all circuit diagrams
    draw_series_circuit()
    draw_parallel_circuit()
    draw_mixed_circuit()
    draw_wheatstone_bridge()
    
    print("All circuit diagrams created.")
    print(f"Images saved to {image_dir}")
