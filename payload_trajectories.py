import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import imageio
import tempfile
from scipy.integrate import solve_ivp

# Create directory for images if it doesn't exist
image_dir = os.path.join('docs', '1 Physics', '2 Gravity', 'images')
os.makedirs(image_dir, exist_ok=True)

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M_EARTH = 5.972e24  # Mass of Earth (kg)
R_EARTH = 6.371e6  # Radius of Earth (m)

# Function to calculate the gravitational acceleration
def gravitational_acceleration(r, M=M_EARTH):
    """Calculate the gravitational acceleration at distance r from a body of mass M.
    
    Args:
        r: Distance from the center of the body (m)
        M: Mass of the body (kg), defaults to Earth's mass
        
    Returns:
        Gravitational acceleration (m/s^2)
    """
    return G * M / r**2

# Function to compute the derivatives for the equations of motion
def payload_dynamics(t, state, mu=G*M_EARTH):
    """Compute the derivatives for the equations of motion of a payload in Earth's gravitational field.
    
    Args:
        t: Time (s)
        state: State vector [x, y, vx, vy]
        mu: Gravitational parameter (G*M) (m^3/s^2)
        
    Returns:
        Derivatives [dx/dt, dy/dt, dvx/dt, dvy/dt]
    """
    x, y, vx, vy = state
    
    # Calculate the distance from Earth's center
    r = np.sqrt(x**2 + y**2)
    
    # Check if the payload has hit Earth's surface
    if r <= R_EARTH:
        return [0, 0, 0, 0]  # Stop the simulation if the payload hits Earth
    
    # Calculate the gravitational acceleration
    a = mu / r**3
    
    # Return the derivatives
    return [vx, vy, -a * x, -a * y]

# Function to simulate the trajectory of a payload
def simulate_trajectory(initial_position, initial_velocity, t_span, t_eval=None):
    """Simulate the trajectory of a payload released near Earth.
    
    Args:
        initial_position: Initial position vector [x, y] (m)
        initial_velocity: Initial velocity vector [vx, vy] (m/s)
        t_span: Time span for the simulation [t_start, t_end] (s)
        t_eval: Times at which to evaluate the solution (s), defaults to None
        
    Returns:
        Solution object from solve_ivp
    """
    # Initial state vector [x, y, vx, vy]
    initial_state = [initial_position[0], initial_position[1], 
                    initial_velocity[0], initial_velocity[1]]
    
    # Solve the initial value problem
    solution = solve_ivp(
        payload_dynamics,
        t_span,
        initial_state,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-8
    )
    
    return solution

# Function to calculate orbital parameters
def calculate_orbital_parameters(position, velocity, mu=G*M_EARTH):
    """Calculate orbital parameters from position and velocity vectors.
    
    Args:
        position: Position vector [x, y] (m)
        velocity: Velocity vector [vx, vy] (m/s)
        mu: Gravitational parameter (G*M) (m^3/s^2)
        
    Returns:
        Dictionary containing orbital parameters
    """
    r = np.sqrt(position[0]**2 + position[1]**2)
    v = np.sqrt(velocity[0]**2 + velocity[1]**2)
    
    # Specific energy
    energy = 0.5 * v**2 - mu / r
    
    # Semi-major axis
    if energy == 0:  # Parabolic orbit
        a = float('inf')
    else:
        a = -mu / (2 * energy)
    
    # Angular momentum per unit mass
    # Use 3D vectors for cross product to avoid deprecation warning
    pos_3d = np.array([position[0], position[1], 0])
    vel_3d = np.array([velocity[0], velocity[1], 0])
    h_vec = np.cross(pos_3d, vel_3d)
    h = np.abs(h_vec[2])  # For 2D, we only care about the z-component
    
    # Eccentricity
    e = np.sqrt(1 + 2 * energy * h**2 / mu**2)
    
    # Orbit type
    if e < 1e-10:  # Numerical tolerance for circular orbit
        orbit_type = "Circular"
    elif e < 1.0:
        orbit_type = "Elliptical"
    elif abs(e - 1.0) < 1e-10:  # Numerical tolerance for parabolic orbit
        orbit_type = "Parabolic"
    else:
        orbit_type = "Hyperbolic"
    
    # Periapsis and apoapsis distances
    if e < 1.0:  # Elliptical orbit
        periapsis = a * (1 - e)
        apoapsis = a * (1 + e)
    elif e == 1.0:  # Parabolic orbit
        periapsis = h**2 / (2 * mu)
        apoapsis = float('inf')
    else:  # Hyperbolic orbit
        periapsis = a * (1 - e)  # Note: a is negative for hyperbolic orbits
        apoapsis = float('inf')
    
    return {
        "semi_major_axis": a,
        "eccentricity": e,
        "specific_energy": energy,
        "angular_momentum": h,
        "orbit_type": orbit_type,
        "periapsis": periapsis,
        "apoapsis": apoapsis
    }

# Function to plot the trajectory of a payload
def plot_trajectory(solution, title="Payload Trajectory", save_path=None):
    """Plot the trajectory of a payload.
    
    Args:
        solution: Solution object from simulate_trajectory
        title: Title of the plot
        save_path: Path to save the plot, defaults to None
        
    Returns:
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Extract the position data
    x = solution.y[0]
    y = solution.y[1]
    
    # Plot the trajectory
    ax.plot(x / R_EARTH, y / R_EARTH, 'b-', label='Payload Trajectory')
    
    # Plot Earth
    earth_circle = plt.Circle((0, 0), 1, color='blue', alpha=0.3, label='Earth')
    ax.add_patch(earth_circle)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Add grid and labels
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('x (Earth radii)', fontsize=12)
    ax.set_ylabel('y (Earth radii)', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Set limits based on the trajectory
    max_dist = max(np.max(np.abs(x)), np.max(np.abs(y))) / R_EARTH
    ax.set_xlim(-max_dist * 1.1, max_dist * 1.1)
    ax.set_ylim(-max_dist * 1.1, max_dist * 1.1)
    
    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

# Function to create an animation of the trajectory
def create_trajectory_animation(solution, title="Payload Trajectory Animation", save_path=None):
    """Create an animation of the trajectory of a payload.
    
    Args:
        solution: Solution object from simulate_trajectory
        title: Title of the animation
        save_path: Path to save the animation, defaults to None
        
    Returns:
        None
    """
    # Create a temporary directory to store frames
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Extract the position data
        x = solution.y[0]
        y = solution.y[1]
        
        # Normalize by Earth's radius
        x_norm = x / R_EARTH
        y_norm = y / R_EARTH
        
        # Calculate the maximum distance for setting the axis limits
        max_dist = max(np.max(np.abs(x_norm)), np.max(np.abs(y_norm)))
        
        # Number of frames
        n_frames = min(100, len(x))  # Limit to 100 frames for efficiency
        indices = np.linspace(0, len(x) - 1, n_frames, dtype=int)
        
        # Generate each frame
        frames_filenames = []
        
        for i, idx in enumerate(indices):
            # Create a new figure for each frame
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Set limits
            ax.set_xlim(-max_dist * 1.1, max_dist * 1.1)
            ax.set_ylim(-max_dist * 1.1, max_dist * 1.1)
            
            # Plot Earth
            earth_circle = plt.Circle((0, 0), 1, color='blue', alpha=0.3, label='Earth')
            ax.add_patch(earth_circle)
            
            # Set equal aspect ratio
            ax.set_aspect('equal')
            
            # Add grid and labels
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xlabel('x (Earth radii)', fontsize=12)
            ax.set_ylabel('y (Earth radii)', fontsize=12)
            ax.set_title(title, fontsize=14)
            
            # Plot the trajectory up to the current point
            ax.plot(x_norm[:idx+1], y_norm[:idx+1], 'b-', label='Trajectory')
            
            # Plot the current position of the payload
            ax.scatter(x_norm[idx], y_norm[idx], color='red', s=50, label='Payload')
            
            # Add legend
            ax.legend(loc='upper right')
            
            # Save the frame
            frame_filename = os.path.join(tmpdirname, f'frame_{i:03d}.png')
            plt.savefig(frame_filename, dpi=100, bbox_inches='tight')
            frames_filenames.append(frame_filename)
            
            # Close the figure to free memory
            plt.close(fig)
        
        # Create the GIF
        if save_path:
            with imageio.get_writer(save_path, mode='I', duration=0.1, loop=0) as writer:
                for frame_filename in frames_filenames:
                    image = imageio.imread(frame_filename)
                    writer.append_data(image)

# Function to plot multiple trajectories for different initial conditions
def plot_multiple_trajectories(initial_conditions, t_span, title="Multiple Payload Trajectories", save_path=None):
    """Plot multiple trajectories for different initial conditions.
    
    Args:
        initial_conditions: List of dictionaries containing initial conditions
        t_span: Time span for the simulation [t_start, t_end] (s)
        title: Title of the plot
        save_path: Path to save the plot, defaults to None
        
    Returns:
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot Earth
    earth_circle = plt.Circle((0, 0), 1, color='blue', alpha=0.3, label='Earth')
    ax.add_patch(earth_circle)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Add grid and labels
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('x (Earth radii)', fontsize=12)
    ax.set_ylabel('y (Earth radii)', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Initialize max_dist for setting axis limits
    max_dist = 1.0  # Start with Earth's radius
    
    # Simulate and plot each trajectory
    for i, condition in enumerate(initial_conditions):
        # Extract the initial conditions
        initial_position = condition['position']
        initial_velocity = condition['velocity']
        label = condition.get('label', f'Trajectory {i+1}')
        color = condition.get('color', None)
        
        # Simulate the trajectory
        solution = simulate_trajectory(initial_position, initial_velocity, t_span)
        
        # Extract the position data
        x = solution.y[0] / R_EARTH
        y = solution.y[1] / R_EARTH
        
        # Update max_dist
        max_dist = max(max_dist, np.max(np.abs(x)), np.max(np.abs(y)))
        
        # Calculate orbital parameters
        params = calculate_orbital_parameters(
            [solution.y[0][0], solution.y[1][0]],
            [solution.y[2][0], solution.y[3][0]]
        )
        
        # Plot the trajectory
        ax.plot(x, y, label=f"{label} ({params['orbit_type']})", color=color)
        
        # Mark the starting point
        ax.scatter(x[0], y[0], color=color if color else 'black', s=50)
    
    # Set limits based on the trajectories
    ax.set_xlim(-max_dist * 1.1, max_dist * 1.1)
    ax.set_ylim(-max_dist * 1.1, max_dist * 1.1)
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

# Function to generate example trajectories
def generate_example_trajectories():
    """Generate example trajectories for different orbital scenarios.
    
    Returns:
        None
    """
    # Common initial position: 300 km above Earth's surface on the x-axis
    altitude = 300e3  # 300 km in meters
    initial_position = [R_EARTH + altitude, 0]  # [x, y] in meters
    
    # Calculate the circular orbit velocity at this altitude
    circular_velocity = np.sqrt(G * M_EARTH / (R_EARTH + altitude))
    
    # 1. Circular orbit
    circular_initial_velocity = [0, circular_velocity]  # [vx, vy] in m/s
    circular_solution = simulate_trajectory(
        initial_position,
        circular_initial_velocity,
        [0, 3 * 3600]  # 3 hours in seconds
    )
    
    # Calculate and print orbital parameters
    circular_params = calculate_orbital_parameters(
        [circular_solution.y[0][0], circular_solution.y[1][0]],
        [circular_solution.y[2][0], circular_solution.y[3][0]]
    )
    print("Circular Orbit Parameters:")
    for key, value in circular_params.items():
        print(f"  {key}: {value}")
    
    # Plot and save the circular orbit trajectory
    plot_trajectory(
        circular_solution,
        title="Circular Orbit Trajectory (300 km altitude)",
        save_path=os.path.join(image_dir, 'circular_orbit.png')
    )
    
    # Create and save an animation of the circular orbit
    create_trajectory_animation(
        circular_solution,
        title="Circular Orbit Animation",
        save_path=os.path.join(image_dir, 'circular_orbit.gif')
    )
    
    # 2. Elliptical orbit
    elliptical_initial_velocity = [0, 0.9 * circular_velocity]  # 90% of circular velocity
    elliptical_solution = simulate_trajectory(
        initial_position,
        elliptical_initial_velocity,
        [0, 3 * 3600]  # 3 hours in seconds
    )
    
    # Calculate and print orbital parameters
    elliptical_params = calculate_orbital_parameters(
        [elliptical_solution.y[0][0], elliptical_solution.y[1][0]],
        [elliptical_solution.y[2][0], elliptical_solution.y[3][0]]
    )
    print("\nElliptical Orbit Parameters:")
    for key, value in elliptical_params.items():
        print(f"  {key}: {value}")
    
    # Plot and save the elliptical orbit trajectory
    plot_trajectory(
        elliptical_solution,
        title="Elliptical Orbit Trajectory (90% of circular velocity)",
        save_path=os.path.join(image_dir, 'elliptical_orbit.png')
    )
    
    # 3. Parabolic trajectory (escape velocity)
    escape_velocity = np.sqrt(2 * G * M_EARTH / (R_EARTH + altitude))
    parabolic_initial_velocity = [0, escape_velocity]  # Escape velocity
    parabolic_solution = simulate_trajectory(
        initial_position,
        parabolic_initial_velocity,
        [0, 6 * 3600]  # 6 hours in seconds
    )
    
    # Calculate and print orbital parameters
    parabolic_params = calculate_orbital_parameters(
        [parabolic_solution.y[0][0], parabolic_solution.y[1][0]],
        [parabolic_solution.y[2][0], parabolic_solution.y[3][0]]
    )
    print("\nParabolic Trajectory Parameters:")
    for key, value in parabolic_params.items():
        print(f"  {key}: {value}")
    
    # Plot and save the parabolic trajectory
    plot_trajectory(
        parabolic_solution,
        title="Parabolic Trajectory (Escape Velocity)",
        save_path=os.path.join(image_dir, 'parabolic_trajectory.png')
    )
    
    # 4. Hyperbolic trajectory
    hyperbolic_initial_velocity = [0, 1.2 * escape_velocity]  # 120% of escape velocity
    hyperbolic_solution = simulate_trajectory(
        initial_position,
        hyperbolic_initial_velocity,
        [0, 6 * 3600]  # 6 hours in seconds
    )
    
    # Calculate and print orbital parameters
    hyperbolic_params = calculate_orbital_parameters(
        [hyperbolic_solution.y[0][0], hyperbolic_solution.y[1][0]],
        [hyperbolic_solution.y[2][0], hyperbolic_solution.y[3][0]]
    )
    print("\nHyperbolic Trajectory Parameters:")
    for key, value in hyperbolic_params.items():
        print(f"  {key}: {value}")
    
    # Plot and save the hyperbolic trajectory
    plot_trajectory(
        hyperbolic_solution,
        title="Hyperbolic Trajectory (120% of Escape Velocity)",
        save_path=os.path.join(image_dir, 'hyperbolic_trajectory.png')
    )
    
    # 5. Reentry trajectory
    reentry_initial_velocity = [0, 0.7 * circular_velocity]  # 70% of circular velocity
    reentry_solution = simulate_trajectory(
        initial_position,
        reentry_initial_velocity,
        [0, 1 * 3600]  # 1 hour in seconds
    )
    
    # Calculate and print orbital parameters
    reentry_params = calculate_orbital_parameters(
        [reentry_solution.y[0][0], reentry_solution.y[1][0]],
        [reentry_solution.y[2][0], reentry_solution.y[3][0]]
    )
    print("\nReentry Trajectory Parameters:")
    for key, value in reentry_params.items():
        print(f"  {key}: {value}")
    
    # Plot and save the reentry trajectory
    plot_trajectory(
        reentry_solution,
        title="Reentry Trajectory (70% of circular velocity)",
        save_path=os.path.join(image_dir, 'reentry_trajectory.png')
    )
    
    # 6. Plot multiple trajectories together
    initial_conditions = [
        {
            'position': initial_position,
            'velocity': [0, 0.7 * circular_velocity],
            'label': 'Reentry (70%)',
            'color': 'red'
        },
        {
            'position': initial_position,
            'velocity': [0, 0.9 * circular_velocity],
            'label': 'Elliptical (90%)',
            'color': 'green'
        },
        {
            'position': initial_position,
            'velocity': circular_initial_velocity,
            'label': 'Circular (100%)',
            'color': 'blue'
        },
        {
            'position': initial_position,
            'velocity': [0, escape_velocity],
            'label': 'Parabolic (Escape)',
            'color': 'purple'
        },
        {
            'position': initial_position,
            'velocity': [0, 1.2 * escape_velocity],
            'label': 'Hyperbolic (120%)',
            'color': 'orange'
        }
    ]
    
    plot_multiple_trajectories(
        initial_conditions,
        [0, 6 * 3600],  # 6 hours in seconds
        title="Comparison of Different Payload Trajectories",
        save_path=os.path.join(image_dir, 'trajectory_comparison.png')
    )
    
    # 7. Trajectories with different release angles
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    angle_conditions = []
    
    for angle in angles:
        # Convert angle to radians
        angle_rad = np.radians(angle)
        
        # Calculate velocity components
        vx = circular_velocity * np.cos(angle_rad)
        vy = circular_velocity * np.sin(angle_rad)
        
        angle_conditions.append({
            'position': initial_position,
            'velocity': [vx, vy],
            'label': f'{angle}°',
            'color': None  # Let matplotlib choose colors
        })
    
    plot_multiple_trajectories(
        angle_conditions,
        [0, 3 * 3600],  # 3 hours in seconds
        title="Payload Trajectories with Different Release Angles (Circular Velocity)",
        save_path=os.path.join(image_dir, 'angle_trajectories.png')
    )
    
    # 8. Trajectories with different initial speeds
    speeds = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    speed_conditions = []
    
    for speed_factor in speeds:
        speed_conditions.append({
            'position': initial_position,
            'velocity': [0, speed_factor * circular_velocity],
            'label': f'{speed_factor:.2f}×V_circ',
            'color': None  # Let matplotlib choose colors
        })
    
    plot_multiple_trajectories(
        speed_conditions,
        [0, 6 * 3600],  # 6 hours in seconds
        title="Payload Trajectories with Different Initial Speeds",
        save_path=os.path.join(image_dir, 'speed_trajectories.png')
    )

# Main function
if __name__ == "__main__":
    # Generate example trajectories
    generate_example_trajectories()
    
    print("\nAll simulations and visualizations completed.")
    print(f"Images saved to {image_dir}")
