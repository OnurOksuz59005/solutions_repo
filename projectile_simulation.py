import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# Create directory for images if it doesn't exist
image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs', '1 Physics', '1 Mechanics', 'images')
os.makedirs(image_dir, exist_ok=True)

def calculate_trajectory(v0, theta_deg, g=9.8, h0=0, time_step=0.01):
    """Calculate the trajectory of a projectile.
    
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
    """Calculate the range of a projectile."""
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

# Parameters
v0 = 20  # m/s
theta_values = np.arange(5, 86, 5)  # degrees
g = 9.8  # m/s²
h0 = 0  # m

# Calculate range for different angles
ranges = [calculate_range(v0, theta, g, h0) for theta in theta_values]

# Find the maximum range and corresponding angle
max_range_idx = np.argmax(ranges)
max_range = ranges[max_range_idx]
optimal_angle = theta_values[max_range_idx]

# Plot range vs angle
plt.figure(figsize=(10, 6))
plt.plot(theta_values, ranges, 'b-', linewidth=2)
plt.plot(optimal_angle, max_range, 'ro', markersize=8)
plt.annotate(f'Maximum Range: {max_range:.2f} m at {optimal_angle}°', 
             xy=(optimal_angle, max_range), xytext=(optimal_angle+5, max_range-5),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
plt.grid(True)
plt.xlabel('Launch Angle (degrees)')
plt.ylabel('Range (m)')
plt.title('Projectile Range vs Launch Angle')
plt.savefig(os.path.join(image_dir, 'range_vs_angle.png'), dpi=300, bbox_inches='tight')

# Plot trajectories for selected angles
plt.figure(figsize=(12, 6))
selected_angles = [15, 30, 45, 60, 75]
colors = ['r', 'g', 'b', 'c', 'm']

for angle, color in zip(selected_angles, colors):
    x, y, _ = calculate_trajectory(v0, angle, g, h0)
    plt.plot(x, y, color=color, linewidth=2, label=f'{angle}°')

plt.grid(True)
plt.xlabel('Horizontal Distance (m)')
plt.ylabel('Height (m)')
plt.title('Projectile Trajectories for Different Launch Angles')
plt.legend()
plt.ylim(bottom=0)
plt.savefig(os.path.join(image_dir, 'trajectories.png'), dpi=300, bbox_inches='tight')

# Investigate how range varies with initial velocity
velocities = np.linspace(10, 50, 5)
plt.figure(figsize=(10, 6))

for v in velocities:
    ranges = [calculate_range(v, theta, g, h0) for theta in theta_values]
    plt.plot(theta_values, ranges, linewidth=2, label=f'v₀ = {v} m/s')

plt.grid(True)
plt.xlabel('Launch Angle (degrees)')
plt.ylabel('Range (m)')
plt.title('Range vs Launch Angle for Different Initial Velocities')
plt.legend()
plt.savefig(os.path.join(image_dir, 'range_vs_velocity.png'), dpi=300, bbox_inches='tight')

# Investigate how range varies with gravity (different planets)
gravities = {
    'Earth': 9.8,
    'Moon': 1.62,
    'Mars': 3.72,
    'Jupiter': 24.79
}

plt.figure(figsize=(10, 6))

for planet, g_value in gravities.items():
    ranges = [calculate_range(v0, theta, g_value, h0) for theta in theta_values]
    plt.plot(theta_values, ranges, linewidth=2, label=f'{planet} (g = {g_value} m/s²)')

plt.grid(True)
plt.xlabel('Launch Angle (degrees)')
plt.ylabel('Range (m)')
plt.title('Range vs Launch Angle on Different Celestial Bodies')
plt.legend()
plt.savefig(os.path.join(image_dir, 'range_vs_gravity.png'), dpi=300, bbox_inches='tight')

print(f"Images saved to {image_dir}")
