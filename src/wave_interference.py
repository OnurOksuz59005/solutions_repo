import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Create directory for images if it doesn't exist
image_dir = os.path.join('docs', '1 Physics', '3 Waves', 'images')
os.makedirs(image_dir, exist_ok=True)

# Constants and parameters
A = 1.0  # Amplitude
lamb = 0.5  # Wavelength (lambda)
f = 1.0  # Frequency
k = 2 * np.pi / lamb  # Wave number
omega = 2 * np.pi * f  # Angular frequency
phi = 0  # Initial phase

# Function to calculate the displacement at a point due to a single source
def calculate_displacement(x, y, t, source_pos):
    """
    Calculate the displacement of the water surface at point (x, y) and time t
    due to a wave from a source at source_pos.
    
    Args:
        x, y: Coordinates of the point
        t: Time
        source_pos: Position of the source (x0, y0)
        
    Returns:
        Displacement at point (x, y) and time t
    """
    x0, y0 = source_pos
    r = np.sqrt((x - x0)**2 + (y - y0)**2)
    # Avoid division by zero at the source
    if r < 1e-10:
        return 0
    return A / np.sqrt(r) * np.cos(k * r - omega * t + phi)

# Function to calculate the total displacement due to multiple sources
def calculate_total_displacement(x, y, t, sources):
    """
    Calculate the total displacement at point (x, y) and time t due to all sources.
    
    Args:
        x, y: Coordinates of the point
        t: Time
        sources: List of source positions [(x1, y1), (x2, y2), ...]
        
    Returns:
        Total displacement at point (x, y) and time t
    """
    total = 0
    for source_pos in sources:
        total += calculate_displacement(x, y, t, source_pos)
    return total

# Function to generate the vertices of a regular polygon
def generate_polygon_vertices(n, radius=1.0, center=(0, 0)):
    """
    Generate the vertices of a regular polygon.
    
    Args:
        n: Number of sides (vertices)
        radius: Distance from center to vertices
        center: Center position (x, y)
        
    Returns:
        List of vertex positions [(x1, y1), (x2, y2), ...]
    """
    vertices = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        vertices.append((x, y))
    return vertices

# Function to create a 2D plot of the interference pattern
def plot_interference_pattern(sources, title, save_path=None, t=0, grid_size=100, plot_range=(-5, 5)):
    """
    Create a 2D plot of the interference pattern.
    
    Args:
        sources: List of source positions [(x1, y1), (x2, y2), ...]
        title: Title of the plot
        save_path: Path to save the plot
        t: Time
        grid_size: Number of points in each dimension
        plot_range: Range of x and y coordinates (min, max)
        
    Returns:
        Figure and axes objects
    """
    x = np.linspace(plot_range[0], plot_range[1], grid_size)
    y = np.linspace(plot_range[0], plot_range[1], grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Calculate the displacement at each point
    Z = np.zeros_like(X)
    for i in range(grid_size):
        for j in range(grid_size):
            Z[i, j] = calculate_total_displacement(X[i, j], Y[i, j], t, sources)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the interference pattern
    im = ax.imshow(Z, extent=[plot_range[0], plot_range[1], plot_range[0], plot_range[1]],
                  cmap='coolwarm', origin='lower', vmin=-3*A, vmax=3*A)
    
    # Add a colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Displacement')
    
    # Mark the positions of the sources
    for i, source in enumerate(sources):
        ax.plot(source[0], source[1], 'ko', markersize=5)
        ax.text(source[0]+0.1, source[1]+0.1, f'S{i+1}', fontsize=10)
    
    # Add grid and labels
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

# Function to create a 3D surface plot of the interference pattern
def plot_3d_interference_pattern(sources, title, save_path=None, t=0, grid_size=100, plot_range=(-5, 5)):
    """
    Create a 3D surface plot of the interference pattern.
    
    Args:
        sources: List of source positions [(x1, y1), (x2, y2), ...]
        title: Title of the plot
        save_path: Path to save the plot
        t: Time
        grid_size: Number of points in each dimension
        plot_range: Range of x and y coordinates (min, max)
        
    Returns:
        Figure and axes objects
    """
    x = np.linspace(plot_range[0], plot_range[1], grid_size)
    y = np.linspace(plot_range[0], plot_range[1], grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Calculate the displacement at each point
    Z = np.zeros_like(X)
    for i in range(grid_size):
        for j in range(grid_size):
            Z[i, j] = calculate_total_displacement(X[i, j], Y[i, j], t, sources)
    
    # Create the plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    
    # Add a colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Displacement')
    
    # Mark the positions of the sources (as vertical lines)
    for i, source in enumerate(sources):
        ax.plot([source[0], source[0]], [source[1], source[1]], [-3*A, 3*A], 'k-', linewidth=1)
        ax.text(source[0], source[1], 3*A, f'S{i+1}', fontsize=10)
    
    # Set labels and title
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('Displacement', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Set z limits
    ax.set_zlim(-3*A, 3*A)
    
    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

# Function to create an animation of the interference pattern
def create_interference_animation(sources, title, save_path=None, duration=2.0, fps=20, grid_size=100, plot_range=(-5, 5)):
    """
    Create an animation of the interference pattern.
    
    Args:
        sources: List of source positions [(x1, y1), (x2, y2), ...]
        title: Title of the animation
        save_path: Path to save the animation
        duration: Duration of the animation in seconds
        fps: Frames per second
        grid_size: Number of points in each dimension
        plot_range: Range of x and y coordinates (min, max)
        
    Returns:
        Animation object
    """
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set up the grid
    x = np.linspace(plot_range[0], plot_range[1], grid_size)
    y = np.linspace(plot_range[0], plot_range[1], grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Calculate the time points
    n_frames = int(duration * fps)
    times = np.linspace(0, duration, n_frames)
    
    # Initialize the plot
    im = ax.imshow(np.zeros((grid_size, grid_size)), extent=[plot_range[0], plot_range[1], plot_range[0], plot_range[1]],
                  cmap='coolwarm', origin='lower', vmin=-3*A, vmax=3*A, animated=True)
    
    # Add a colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Displacement')
    
    # Mark the positions of the sources
    for i, source in enumerate(sources):
        ax.plot(source[0], source[1], 'ko', markersize=5)
        ax.text(source[0]+0.1, source[1]+0.1, f'S{i+1}', fontsize=10)
    
    # Add grid and labels
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Function to update the plot for each frame
    def update(frame):
        t = times[frame]
        Z = np.zeros_like(X)
        for i in range(grid_size):
            for j in range(grid_size):
                Z[i, j] = calculate_total_displacement(X[i, j], Y[i, j], t, sources)
        im.set_array(Z)
        return [im]
    
    # Create the animation
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=True)
    
    # Save the animation if a save path is provided
    if save_path:
        anim.save(save_path, writer='pillow', fps=fps)
    
    return anim

# Function to analyze different polygon configurations
def analyze_polygon_interference(n_vertices_list=[3, 4, 5, 6], radius=3.0, plot_range=(-5, 5)):
    """
    Analyze interference patterns for different regular polygons.
    
    Args:
        n_vertices_list: List of numbers of vertices to analyze
        radius: Radius of the polygon
        plot_range: Range of x and y coordinates (min, max)
        
    Returns:
        None
    """
    for n_vertices in n_vertices_list:
        # Generate the vertices of the polygon
        sources = generate_polygon_vertices(n_vertices, radius=radius)
        
        # Create a descriptive name for the polygon
        if n_vertices == 3:
            polygon_name = "Triangle"
        elif n_vertices == 4:
            polygon_name = "Square"
        elif n_vertices == 5:
            polygon_name = "Pentagon"
        elif n_vertices == 6:
            polygon_name = "Hexagon"
        else:
            polygon_name = f"{n_vertices}-gon"
        
        # Plot the 2D interference pattern
        plot_interference_pattern(
            sources,
            f"Interference Pattern for {polygon_name}",
            save_path=os.path.join(image_dir, f"{polygon_name.lower()}_interference_2d.png"),
            grid_size=200
        )
        
        # Plot the 3D interference pattern
        plot_3d_interference_pattern(
            sources,
            f"3D Interference Pattern for {polygon_name}",
            save_path=os.path.join(image_dir, f"{polygon_name.lower()}_interference_3d.png"),
            grid_size=100
        )
        
        # Create an animation of the interference pattern
        create_interference_animation(
            sources,
            f"Interference Animation for {polygon_name}",
            save_path=os.path.join(image_dir, f"{polygon_name.lower()}_interference_animation.gif"),
            grid_size=100
        )

# Main function
if __name__ == "__main__":
    # Analyze interference patterns for different polygons
    analyze_polygon_interference()
    
    print("All simulations and visualizations completed.")
    print(f"Images saved to {image_dir}")
