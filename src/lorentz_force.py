import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os

# Create directory for images if it doesn't exist
image_dir = os.path.join('docs', '1 Physics', '4 Electromagnetism', 'images')
os.makedirs(image_dir, exist_ok=True)

def lorentz_force(q, E, v, B):
    """
    Calculate the Lorentz force on a charged particle.
    
    Args:
        q (float): Charge of the particle in Coulombs
        E (numpy.ndarray): Electric field vector in N/C
        v (numpy.ndarray): Velocity vector of the particle in m/s
        B (numpy.ndarray): Magnetic field vector in Tesla
        
    Returns:
        numpy.ndarray: Force vector in Newtons
    """
    return q * (E + np.cross(v, B))

def acceleration(q, m, E, v, B):
    """
    Calculate the acceleration of a charged particle due to the Lorentz force.
    
    Args:
        q (float): Charge of the particle in Coulombs
        m (float): Mass of the particle in kg
        E (numpy.ndarray): Electric field vector in N/C
        v (numpy.ndarray): Velocity vector of the particle in m/s
        B (numpy.ndarray): Magnetic field vector in Tesla
        
    Returns:
        numpy.ndarray: Acceleration vector in m/s²
    """
    F = lorentz_force(q, E, v, B)
    return F / m

def runge_kutta_step(q, m, E, v, r, B, dt):
    """
    Perform a single step of the 4th-order Runge-Kutta method to update position and velocity.
    
    Args:
        q (float): Charge of the particle in Coulombs
        m (float): Mass of the particle in kg
        E (numpy.ndarray): Electric field vector in N/C
        v (numpy.ndarray): Current velocity vector in m/s
        r (numpy.ndarray): Current position vector in m
        B (numpy.ndarray): Magnetic field vector in Tesla
        dt (float): Time step in seconds
        
    Returns:
        tuple: Updated position and velocity vectors
    """
    # Calculate k1
    a1 = acceleration(q, m, E, v, B)
    k1_v = a1 * dt
    k1_r = v * dt
    
    # Calculate k2
    a2 = acceleration(q, m, E, v + 0.5 * k1_v, B)
    k2_v = a2 * dt
    k2_r = (v + 0.5 * k1_v) * dt
    
    # Calculate k3
    a3 = acceleration(q, m, E, v + 0.5 * k2_v, B)
    k3_v = a3 * dt
    k3_r = (v + 0.5 * k2_v) * dt
    
    # Calculate k4
    a4 = acceleration(q, m, E, v + k3_v, B)
    k4_v = a4 * dt
    k4_r = (v + k3_v) * dt
    
    # Update velocity and position
    v_new = v + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
    r_new = r + (k1_r + 2*k2_r + 2*k3_r + k4_r) / 6
    
    return r_new, v_new

def simulate_particle_motion(q, m, E, B, v0, r0, dt, steps):
    """
    Simulate the motion of a charged particle in electric and magnetic fields.
    
    Args:
        q (float): Charge of the particle in Coulombs
        m (float): Mass of the particle in kg
        E (numpy.ndarray): Electric field vector in N/C
        B (numpy.ndarray): Magnetic field vector in Tesla
        v0 (numpy.ndarray): Initial velocity vector in m/s
        r0 (numpy.ndarray): Initial position vector in m
        dt (float): Time step in seconds
        steps (int): Number of simulation steps
        
    Returns:
        tuple: Arrays of positions and velocities over time
    """
    # Initialize arrays to store positions and velocities
    positions = np.zeros((steps, 3))
    velocities = np.zeros((steps, 3))
    
    # Set initial conditions
    positions[0] = r0
    velocities[0] = v0
    
    # Perform simulation using Runge-Kutta method
    for i in range(1, steps):
        positions[i], velocities[i] = runge_kutta_step(
            q, m, E, velocities[i-1], positions[i-1], B, dt
        )
    
    return positions, velocities

def calculate_larmor_radius(q, m, v_perp, B_mag):
    """
    Calculate the Larmor radius (gyroradius) of a charged particle in a magnetic field.
    
    Args:
        q (float): Charge of the particle in Coulombs
        m (float): Mass of the particle in kg
        v_perp (float): Velocity component perpendicular to the magnetic field in m/s
        B_mag (float): Magnitude of the magnetic field in Tesla
        
    Returns:
        float: Larmor radius in meters
    """
    return m * v_perp / (abs(q) * B_mag)

def calculate_cyclotron_frequency(q, m, B_mag):
    """
    Calculate the cyclotron frequency of a charged particle in a magnetic field.
    
    Args:
        q (float): Charge of the particle in Coulombs
        m (float): Mass of the particle in kg
        B_mag (float): Magnitude of the magnetic field in Tesla
        
    Returns:
        float: Cyclotron frequency in radians per second
    """
    return abs(q) * B_mag / m

def calculate_drift_velocity(q, E, B):
    """
    Calculate the E×B drift velocity of a charged particle.
    
    Args:
        q (float): Charge of the particle in Coulombs (not used in calculation but included for completeness)
        E (numpy.ndarray): Electric field vector in N/C
        B (numpy.ndarray): Magnetic field vector in Tesla
        
    Returns:
        numpy.ndarray: Drift velocity vector in m/s
    """
    B_squared = np.sum(B**2)
    return np.cross(E, B) / B_squared

def plot_trajectory_2d(positions, title, save_path=None, E=None, B=None):
    """
    Plot the 2D projection of a particle's trajectory.
    
    Args:
        positions (numpy.ndarray): Array of particle positions
        title (str): Plot title
        save_path (str, optional): Path to save the plot image
        E (numpy.ndarray, optional): Electric field vector
        B (numpy.ndarray, optional): Magnetic field vector
    """
    plt.figure(figsize=(10, 8))
    plt.plot(positions[:, 0], positions[:, 1], 'b-', label='Trajectory')
    plt.plot(positions[0, 0], positions[0, 1], 'go', label='Start')
    plt.plot(positions[-1, 0], positions[-1, 1], 'ro', label='End')
    
    # Calculate axis limits for positioning field indicators
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Add field vectors if provided - place in corners
    if E is not None and np.any(E):
        # Plot electric field vector in top-right corner
        corner_x = x_max + 0.05 * x_range
        corner_y = y_max - 0.05 * y_range
        E_scale = 0.15 * max(x_range, y_range)
        
        # For 2D plot, only show x and y components
        plt.arrow(corner_x, corner_y, E_scale * E[0], E_scale * E[1], 
                 color='r', width=0.005 * E_scale, head_width=0.03 * E_scale, 
                 head_length=0.05 * E_scale, label='E field')
    
    if B is not None and np.any(B):
        # Plot magnetic field vector in top-left corner
        corner_x = x_min - 0.05 * x_range
        corner_y = y_max - 0.05 * y_range
        
        # For 2D plot, we represent B field with a circle and dot/cross
        if B[2] > 0:  # B field coming out of the plane
            plt.scatter(corner_x, corner_y, color='g', marker='o', s=300, label='B field (out)')
            plt.scatter(corner_x, corner_y, color='g', marker='.', s=100)
        elif B[2] < 0:  # B field going into the plane
            plt.scatter(corner_x, corner_y, color='g', marker='o', s=300, label='B field (in)')
            plt.scatter(corner_x, corner_y, color='g', marker='x', s=100)
        else:  # B field in the plane
            B_scale = 0.15 * max(x_range, y_range)
            plt.arrow(corner_x, corner_y, B_scale * B[0], B_scale * B[1], 
                     color='g', width=0.005 * B_scale, head_width=0.03 * B_scale, 
                     head_length=0.05 * B_scale, label='B field')
    
    plt.title(title)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True)
    
    # Move legend to upper left corner
    plt.legend(loc='upper left', fontsize='small')
    
    # Adjust axis limits to show more context
    plt.xlim(x_min - 0.2 * x_range, x_max + 0.2 * x_range)
    plt.ylim(y_min - 0.2 * y_range, y_max + 0.2 * y_range)
    
    # Format tick labels
    plt.ticklabel_format(style='sci', scilimits=(-2, 2), axis='both')
    
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 2D plot to {save_path}")
    
    plt.show()

def plot_trajectory_3d(positions, title, save_path=None, E=None, B=None):
    """
    Plot the 3D trajectory of a particle.
    
    Args:
        positions (numpy.ndarray): Array of particle positions
        title (str): Plot title
        save_path (str, optional): Path to save the plot image
        E (numpy.ndarray, optional): Electric field vector
        B (numpy.ndarray, optional): Magnetic field vector
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', label='Trajectory')
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='g', s=100, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='r', s=100, label='End')
    
    # Calculate axis limits for positioning field indicators
    max_range = np.array([positions[:, 0].max() - positions[:, 0].min(),
                          positions[:, 1].max() - positions[:, 1].min(),
                          positions[:, 2].max() - positions[:, 2].min()]).max() / 2.0
    
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    
    x_min, x_max = mid_x - max_range, mid_x + max_range
    y_min, y_max = mid_y - max_range, mid_y + max_range
    z_min, z_max = mid_z - max_range, mid_z + max_range
    
    # Add field vectors if provided - place in corners
    if E is not None and np.any(E):
        # Plot electric field vector in top corner
        corner_x = x_max - 0.1 * (x_max - x_min)
        corner_y = y_max - 0.1 * (y_max - y_min)
        corner_z = z_max - 0.1 * (z_max - z_min)
        
        E_scale = 0.15 * max_range
        ax.quiver(corner_x, corner_y, corner_z, 
                 E_scale * E[0], E_scale * E[1], E_scale * E[2], 
                 color='r', arrow_length_ratio=0.3, label='E field')
    
    if B is not None and np.any(B):
        # Plot magnetic field vector in different corner
        corner_x = x_min + 0.1 * (x_max - x_min)
        corner_y = y_max - 0.1 * (y_max - y_min)
        corner_z = z_max - 0.1 * (z_max - z_min)
        
        B_scale = 0.15 * max_range
        ax.quiver(corner_x, corner_y, corner_z, 
                 B_scale * B[0], B_scale * B[1], B_scale * B[2], 
                 color='g', arrow_length_ratio=0.3, label='B field')
    
    ax.set_title(title)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # Move legend to upper left corner
    ax.legend(loc='upper left', fontsize='small')
    
    # Set axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    # Format tick labels
    ax.ticklabel_format(style='sci', scilimits=(-2, 2), axis='both')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 3D plot to {save_path}")
    
    plt.show()

def create_animation(positions, title, save_path=None, E=None, B=None):
    """
    Create an animation of the particle's trajectory.
    
    Args:
        positions (numpy.ndarray): Array of particle positions
        title (str): Animation title
        save_path (str, optional): Path to save the animation
        E (numpy.ndarray, optional): Electric field vector
        B (numpy.ndarray, optional): Magnetic field vector
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up the initial plot
    line, = ax.plot([], [], [], 'b-', label='Trajectory')
    point, = ax.plot([], [], [], 'ro', markersize=8)
    
    # Calculate axis limits first to position field indicators properly
    max_range = np.array([positions[:, 0].max() - positions[:, 0].min(),
                          positions[:, 1].max() - positions[:, 1].min(),
                          positions[:, 2].max() - positions[:, 2].min()]).max() / 2.0
    
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    
    x_min, x_max = mid_x - max_range, mid_x + max_range
    y_min, y_max = mid_y - max_range, mid_y + max_range
    z_min, z_max = mid_z - max_range, mid_z + max_range
    
    # Add field vectors if provided - place in corners
    if E is not None and np.any(E):
        # Plot electric field vector in top corner
        corner_x = x_max - 0.1 * (x_max - x_min)
        corner_y = y_max - 0.1 * (y_max - y_min)
        corner_z = z_max - 0.1 * (z_max - z_min)
        
        E_scale = 0.15 * max_range
        ax.quiver(corner_x, corner_y, corner_z, 
                 E_scale * E[0], E_scale * E[1], E_scale * E[2], 
                 color='r', arrow_length_ratio=0.3, label='E field')
    
    if B is not None and np.any(B):
        # Plot magnetic field vector in different corner
        corner_x = x_min + 0.1 * (x_max - x_min)
        corner_y = y_max - 0.1 * (y_max - y_min)
        corner_z = z_max - 0.1 * (z_max - z_min)
        
        B_scale = 0.15 * max_range
        ax.quiver(corner_x, corner_y, corner_z, 
                 B_scale * B[0], B_scale * B[1], B_scale * B[2], 
                 color='g', arrow_length_ratio=0.3, label='B field')
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    
    # Set axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    # Move legend to upper left corner
    ax.legend(loc='upper left', fontsize='small')
    
    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        return line, point
    
    def animate(i):
        # Use a window of points to show the trajectory
        window_size = 50
        start_idx = max(0, i - window_size)
        
        x = positions[start_idx:i+1, 0]
        y = positions[start_idx:i+1, 1]
        z = positions[start_idx:i+1, 2]
        
        line.set_data(x, y)
        line.set_3d_properties(z)
        
        point.set_data([positions[i, 0]], [positions[i, 1]])
        point.set_3d_properties([positions[i, 2]])
        
        return line, point
    
    # Create animation
    frames = min(len(positions), 200)  # Limit frames for performance
    step = len(positions) // frames if len(positions) > frames else 1
    ani = FuncAnimation(fig, animate, frames=range(0, len(positions), step),
                        init_func=init, blit=True, interval=50)
    
    if save_path:
        ani.save(save_path, writer='pillow', fps=20)
        print(f"Saved animation to {save_path}")
    
    plt.show()

# Example usage
if __name__ == "__main__":
    # Electron properties
    q_electron = -1.602e-19  # Charge in Coulombs
    m_electron = 9.109e-31   # Mass in kg
    
    # Scenario 1: Uniform Magnetic Field
    def run_uniform_magnetic_field_scenario():
        # Field configuration
        B = np.array([0, 0, 1.0])  # Magnetic field in Tesla (z-direction)
        E = np.array([0, 0, 0])    # No electric field
        
        # Initial conditions
        v0 = np.array([1e6, 1e6, 1e5])  # Initial velocity in m/s
        r0 = np.array([0, 0, 0])        # Initial position at origin
        
        # Simulation parameters
        dt = 1e-12  # Time step in seconds
        steps = 1000  # Number of steps
        
        # Run simulation
        positions, velocities = simulate_particle_motion(
            q_electron, m_electron, E, B, v0, r0, dt, steps
        )
        
        # Calculate theoretical values
        v_perp = np.sqrt(v0[0]**2 + v0[1]**2)  # Perpendicular velocity component
        B_mag = np.linalg.norm(B)  # Magnitude of magnetic field
        
        larmor_radius = calculate_larmor_radius(q_electron, m_electron, v_perp, B_mag)
        cyclotron_freq = calculate_cyclotron_frequency(q_electron, m_electron, B_mag)
        
        print(f"Theoretical Larmor radius: {larmor_radius:.6e} m")
        print(f"Theoretical cyclotron frequency: {cyclotron_freq:.6e} rad/s")
        
        # Plot trajectories
        plot_trajectory_2d(
            positions, 
            f"Electron in Uniform Magnetic Field (B = {B_mag} T)",
            os.path.join(image_dir, 'uniform_magnetic_field_2d.png'),
            E=E, B=B
        )
        
        plot_trajectory_3d(
            positions, 
            f"Electron in Uniform Magnetic Field (B = {B_mag} T)",
            os.path.join(image_dir, 'uniform_magnetic_field_3d.png'),
            E=E, B=B
        )
        
        # Create animation
        create_animation(
            positions,
            f"Electron in Uniform Magnetic Field (B = {B_mag} T)",
            os.path.join(image_dir, 'uniform_magnetic_field_animation.gif'),
            E=E, B=B
        )
    
    # Scenario 2: Combined Electric and Magnetic Fields
    def run_combined_fields_scenario():
        # Field configuration
        B = np.array([0, 0, 1.0])      # Magnetic field in Tesla (z-direction)
        E = np.array([1e5, 0, 0])      # Electric field in N/C (x-direction)
        
        # Initial conditions
        v0 = np.array([0, 0, 0])       # Starting from rest
        r0 = np.array([0, 0, 0])       # Initial position at origin
        
        # Simulation parameters
        dt = 1e-12  # Time step in seconds
        steps = 2000  # Number of steps
        
        # Run simulation
        positions, velocities = simulate_particle_motion(
            q_electron, m_electron, E, B, v0, r0, dt, steps
        )
        
        # Calculate theoretical drift velocity
        drift_v = calculate_drift_velocity(q_electron, E, B)
        print(f"Theoretical E×B drift velocity: {drift_v} m/s")
        
        # Plot trajectories
        plot_trajectory_2d(
            positions, 
            f"Electron in Combined E and B Fields",
            os.path.join(image_dir, 'combined_fields_2d.png'),
            E=E, B=B
        )
        
        plot_trajectory_3d(
            positions, 
            f"Electron in Combined E and B Fields",
            os.path.join(image_dir, 'combined_fields_3d.png'),
            E=E, B=B
        )
        
        # Create animation
        create_animation(
            positions,
            f"Electron in Combined E and B Fields",
            os.path.join(image_dir, 'combined_fields_animation.gif'),
            E=E, B=B
        )
    
    # Scenario 3: Crossed Electric and Magnetic Fields
    def run_crossed_fields_scenario():
        # Field configuration
        B = np.array([0, 0, 1.0])      # Magnetic field in Tesla (z-direction)
        E = np.array([0, 1e5, 0])      # Electric field in N/C (y-direction)
        
        # Initial conditions
        v0 = np.array([1e6, 0, 0])     # Initial velocity in x-direction
        r0 = np.array([0, 0, 0])       # Initial position at origin
        
        # Simulation parameters
        dt = 1e-12  # Time step in seconds
        steps = 2000  # Number of steps
        
        # Run simulation
        positions, velocities = simulate_particle_motion(
            q_electron, m_electron, E, B, v0, r0, dt, steps
        )
        
        # Plot trajectories
        plot_trajectory_2d(
            positions, 
            f"Electron in Crossed E and B Fields",
            os.path.join(image_dir, 'crossed_fields_2d.png'),
            E=E, B=B
        )
        
        plot_trajectory_3d(
            positions, 
            f"Electron in Crossed E and B Fields",
            os.path.join(image_dir, 'crossed_fields_3d.png'),
            E=E, B=B
        )
        
        # Create animation
        create_animation(
            positions,
            f"Electron in Crossed E and B Fields",
            os.path.join(image_dir, 'crossed_fields_animation.gif'),
            E=E, B=B
        )
    
    # Parameter Exploration
    def explore_magnetic_field_strength():
        # Field configuration
        B_strengths = [0.5, 1.0, 2.0]  # Different magnetic field strengths in Tesla
        E = np.array([0, 0, 0])        # No electric field
        
        # Initial conditions
        v0 = np.array([1e6, 1e6, 1e5])  # Initial velocity in m/s
        r0 = np.array([0, 0, 0])        # Initial position at origin
        
        # Simulation parameters
        dt = 1e-12  # Time step in seconds
        steps = 1000  # Number of steps
        
        # Run simulations for different magnetic field strengths
        fig, axs = plt.subplots(1, len(B_strengths), figsize=(15, 5))
        
        for i, B_mag in enumerate(B_strengths):
            B = np.array([0, 0, B_mag])  # Magnetic field in z-direction
            
            # Run simulation
            positions, velocities = simulate_particle_motion(
                q_electron, m_electron, E, B, v0, r0, dt, steps
            )
            
            # Plot trajectory
            ax = axs[i]
            ax.plot(positions[:, 0], positions[:, 1], label=f'B = {B_mag} T')
            ax.plot(positions[0, 0], positions[0, 1], 'go', label='Start')
            ax.plot(positions[-1, 0], positions[-1, 1], 'ro', label='End')
            
            # Add magnetic field indicator - move to corner
            x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
            y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            # Position in top-right corner
            corner_x = x_max + 0.05 * x_range
            corner_y = y_max - 0.05 * y_range
            ax.scatter(corner_x, corner_y, color='g', marker='o', s=300, label='B field (out)')
            ax.scatter(corner_x, corner_y, color='g', marker='.', s=100)
            
            # Improve axis formatting
            ax.set_title(f'B = {B_mag} T')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.grid(True)
            
            # Move legend to upper left corner
            ax.legend(loc='upper left', fontsize='small')
            
            # Adjust axis limits to show more context around the trajectory
            ax.set_xlim(x_min - 0.2 * x_range, x_max + 0.2 * x_range)
            ax.set_ylim(y_min - 0.2 * y_range, y_max + 0.2 * y_range)
            
            # Format the tick labels to be more readable
            ax.ticklabel_format(style='sci', scilimits=(-2, 2), axis='both')
            
            # Ensure equal aspect ratio
            ax.set_aspect('equal')
        
        plt.suptitle('Effect of Magnetic Field Strength on Electron Trajectory')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save the plot
        save_path = os.path.join(image_dir, 'magnetic_field_strength_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved magnetic field strength comparison to {save_path}")
        
        plt.show()
    
    def explore_particle_mass():
        # Field configuration
        B = np.array([0, 0, 1.0])  # Magnetic field in Tesla (z-direction)
        E = np.array([0, 0, 0])    # No electric field
        
        # Initial conditions
        v0 = np.array([1e6, 1e6, 1e5])  # Initial velocity in m/s
        r0 = np.array([0, 0, 0])        # Initial position at origin
        
        # Simulation parameters
        dt = 1e-12  # Time step in seconds
        steps = 1000  # Number of steps
        
        # Different particle masses (relative to electron mass)
        mass_factors = [1, 10, 1836]  # electron, fictional particle, proton
        particle_names = ['Electron', 'Ion (10m_e)', 'Proton']
        
        # Run simulations for different particle masses
        fig, axs = plt.subplots(1, len(mass_factors), figsize=(15, 5))
        
        for i, (factor, name) in enumerate(zip(mass_factors, particle_names)):
            m = m_electron * factor
            
            # Run simulation
            positions, velocities = simulate_particle_motion(
                q_electron, m, E, B, v0, r0, dt, steps
            )
            
            # Plot trajectory
            ax = axs[i]
            ax.plot(positions[:, 0], positions[:, 1], label=name)
            ax.plot(positions[0, 0], positions[0, 1], 'go', label='Start')
            ax.plot(positions[-1, 0], positions[-1, 1], 'ro', label='End')
            
            # Add magnetic field indicator - move to corner
            x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
            y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            # Position in top-right corner
            corner_x = x_max + 0.05 * x_range
            corner_y = y_max - 0.05 * y_range
            ax.scatter(corner_x, corner_y, color='g', marker='o', s=300, label='B field (out)')
            ax.scatter(corner_x, corner_y, color='g', marker='.', s=100)
            
            # Improve axis formatting
            ax.set_title(name)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.grid(True)
            
            # Move legend to upper left corner
            ax.legend(loc='upper left', fontsize='small')
            
            # Adjust axis limits to show more context around the trajectory
            ax.set_xlim(x_min - 0.2 * x_range, x_max + 0.2 * x_range)
            ax.set_ylim(y_min - 0.2 * y_range, y_max + 0.2 * y_range)
            
            # Format the tick labels to be more readable
            ax.ticklabel_format(style='sci', scilimits=(-2, 2), axis='both')
            
            # Ensure equal aspect ratio
            ax.set_aspect('equal')
        
        plt.suptitle('Effect of Particle Mass on Trajectory in Magnetic Field')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save the plot
        save_path = os.path.join(image_dir, 'particle_mass_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved particle mass comparison to {save_path}")
        
        plt.show()
    
    # Run all scenarios
    print("Running Uniform Magnetic Field Scenario...")
    run_uniform_magnetic_field_scenario()
    
    print("\nRunning Combined Fields Scenario...")
    run_combined_fields_scenario()
    
    print("\nRunning Crossed Fields Scenario...")
    run_crossed_fields_scenario()
    
    print("\nExploring Magnetic Field Strength...")
    explore_magnetic_field_strength()
    
    print("\nExploring Particle Mass...")
    explore_particle_mass()
