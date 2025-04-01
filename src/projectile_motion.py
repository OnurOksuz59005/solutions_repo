import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation

# Create the images directory if it doesn't exist
output_dir = os.path.join('docs', '1 Physics', '1 Mechanics', 'images')
os.makedirs(output_dir, exist_ok=True)

def calculate_trajectory(v0, theta_deg, g=9.8, h0=0, time_step=0.01):
    """
    Calculate the trajectory of a projectile.
    
    Args:
        v0: Initial velocity (m/s)
        theta_deg: Launch angle (degrees)
        g: Gravitational acceleration (m/s²)
        h0: Initial height (m)
        time_step: Time step for simulation (s)
    
    Returns:
        Tuple of (x_positions, y_positions, time_of_flight)
    """
    # Convert angle to radians
    theta = np.radians(theta_deg)
    
    # Initial velocity components
    v0x = v0 * np.cos(theta)
    v0y = v0 * np.sin(theta)
    
    # Time of flight (solve for when y = 0)
    # Using quadratic formula: h0 + v0y*t - 0.5*g*t² = 0
    discriminant = v0y**2 + 2*g*h0
    if discriminant < 0:  # No real solutions
        return [], [], 0
    
    t_flight = (v0y + np.sqrt(discriminant)) / g
    
    # Generate time points
    t = np.arange(0, t_flight + time_step, time_step)
    
    # Calculate positions
    x = v0x * t
    y = h0 + v0y * t - 0.5 * g * t**2
    
    return x, y, t_flight

def calculate_range(v0, theta_deg, g=9.8, h0=0):
    """
    Calculate the range of a projectile.
    """
    x, y, _ = calculate_trajectory(v0, theta_deg, g, h0)
    if len(x) > 0:
        # Find the index where y becomes negative
        landing_idx = np.where(y < 0)[0]
        if len(landing_idx) > 0:
            idx = landing_idx[0]
            # Linear interpolation to find exact landing point
            if idx > 0:
                x_range = x[idx-1] + (x[idx] - x[idx-1]) * (-y[idx-1]) / (y[idx] - y[idx-1])
                return x_range
        return x[-1]  # If no negative y, return the last x
    return 0

def generate_range_vs_angle_plot():
    """
    Generate a plot showing the relationship between range and launch angle.
    """
    # Parameters
    v0 = 20  # m/s
    theta_values = np.arange(0, 91, 1)  # degrees
    g = 9.8  # m/s²
    h0 = 0  # m

    # Calculate range for different angles
    ranges = [calculate_range(v0, theta, g, h0) for theta in theta_values]

    # Find the maximum range and corresponding angle
    max_range_idx = np.argmax(ranges)
    max_range = ranges[max_range_idx]
    optimal_angle = theta_values[max_range_idx]

    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot range vs angle
    plt.plot(theta_values, ranges, 'b-', linewidth=2)
    plt.plot(optimal_angle, max_range, 'ro', markersize=8)
    
    # Add annotation for maximum range
    plt.annotate(f'Maximum Range: {max_range:.2f} m at {optimal_angle}°', 
                xy=(optimal_angle, max_range), 
                xytext=(optimal_angle+10, max_range-5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # Add theoretical curve
    theoretical_angles = np.arange(0, 91, 1)
    theoretical_ranges = (v0**2 * np.sin(2 * np.radians(theoretical_angles))) / g
    plt.plot(theoretical_angles, theoretical_ranges, 'g--', linewidth=1.5, 
             label='Theoretical: $R = \\frac{v_0^2 \\sin(2\\theta)}{g}$')
    
    # Add grid, labels, and title
    plt.grid(True, alpha=0.3)
    plt.xlabel('Launch Angle (degrees)', fontsize=12)
    plt.ylabel('Range (m)', fontsize=12)
    plt.title('Projectile Range vs Launch Angle', fontsize=14)
    plt.legend()
    
    # Add text box with parameters
    textstr = f'Initial velocity: {v0} m/s\nGravitational acceleration: {g} m/s²'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'range_vs_angle.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_trajectory_comparison():
    """
    Generate a plot comparing trajectories for different launch angles.
    """
    # Parameters
    v0 = 20  # m/s
    selected_angles = [15, 30, 45, 60, 75]
    colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A8', '#33A8FF']
    g = 9.8  # m/s²
    h0 = 0  # m

    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Maximum range for scaling
    max_range = calculate_range(v0, 45, g, h0)
    
    # Plot trajectories for selected angles
    for angle, color in zip(selected_angles, colors):
        x, y, _ = calculate_trajectory(v0, angle, g, h0)
        plt.plot(x, y, color=color, linewidth=2, label=f'{angle}°')
        
        # Mark the landing point
        range_val = calculate_range(v0, angle, g, h0)
        plt.plot(range_val, 0, 'o', color=color, markersize=6)
        
        # Add range annotation
        plt.annotate(f'{range_val:.1f} m', 
                    xy=(range_val, 0), 
                    xytext=(range_val, -1),
                    ha='center',
                    fontsize=8)
    
    # Add grid, labels, and title
    plt.grid(True, alpha=0.3)
    plt.xlabel('Horizontal Distance (m)', fontsize=12)
    plt.ylabel('Height (m)', fontsize=12)
    plt.title('Projectile Trajectories for Different Launch Angles', fontsize=14)
    plt.legend(loc='upper right')
    
    # Set axis limits
    plt.xlim(0, max_range * 1.05)
    plt.ylim(0, max_range * 0.5)
    
    # Add text box with parameters
    textstr = f'Initial velocity: {v0} m/s\nGravitational acceleration: {g} m/s²'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'trajectory_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_parameter_effects():
    """
    Generate a plot showing the effects of initial velocity and gravity on range.
    """
    # Parameters
    theta = 45  # degrees (optimal angle)
    v0_values = np.arange(10, 31, 5)  # m/s
    g_values = [9.8, 3.7, 1.6]  # m/s² (Earth, Mars, Moon)
    g_labels = ['Earth (g = 9.8 m/s²)', 'Mars (g = 3.7 m/s²)', 'Moon (g = 1.6 m/s²)']
    colors = ['blue', 'red', 'green']
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot range vs initial velocity for different g values
    for g, label, color in zip(g_values, g_labels, colors):
        ranges = [(v0**2 * np.sin(2 * np.radians(theta))) / g for v0 in v0_values]
        plt.plot(v0_values, ranges, 'o-', color=color, linewidth=2, label=label)
    
    # Add grid, labels, and title
    plt.grid(True, alpha=0.3)
    plt.xlabel('Initial Velocity (m/s)', fontsize=12)
    plt.ylabel('Range (m)', fontsize=12)
    plt.title('Effect of Initial Velocity and Gravity on Projectile Range', fontsize=14)
    plt.legend()
    
    # Add text box with formula
    textstr = 'Range Formula:\n$R = \\frac{v_0^2 \\sin(2\\theta)}{g}$\nAt θ = 45°:\n$R = \\frac{v_0^2}{g}$'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'parameter_effects.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Generating projectile motion visualizations...")
    
    # Generate all visualizations
    generate_range_vs_angle_plot()
    print("Created range vs angle plot")
    
    generate_trajectory_comparison()
    print("Created trajectory comparison plot")
    
    generate_parameter_effects()
    print("Created parameter effects plot")
    
    print(f"\nAll visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
