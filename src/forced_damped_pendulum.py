import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap

# Create the images directory if it doesn't exist
output_dir = os.path.join('docs', '1 Physics', '1 Mechanics', 'images')
os.makedirs(output_dir, exist_ok=True)

def pendulum_system(t, y, b, A, omega_d, omega_0_sq):
    """
    System of first-order ODEs for the forced damped pendulum.
    
    Args:
        t: Time
        y: State vector [theta, omega]
        b: Damping coefficient
        A: Driving amplitude
        omega_d: Driving frequency
        omega_0_sq: Natural frequency squared (g/L)
    
    Returns:
        Derivatives [dtheta/dt, domega/dt]
    """
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -omega_0_sq * np.sin(theta) - b * omega + A * np.cos(omega_d * t)
    return [dtheta_dt, domega_dt]

def linear_pendulum_system(t, y, b, A, omega_d, omega_0_sq):
    """
    System of first-order ODEs for the linearized forced damped pendulum.
    
    Args:
        t: Time
        y: State vector [theta, omega]
        b: Damping coefficient
        A: Driving amplitude
        omega_d: Driving frequency
        omega_0_sq: Natural frequency squared (g/L)
    
    Returns:
        Derivatives [dtheta/dt, domega/dt]
    """
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -omega_0_sq * theta - b * omega + A * np.cos(omega_d * t)
    return [dtheta_dt, domega_dt]

def generate_damping_effect():
    """
    Generate a plot showing the effect of damping coefficient on pendulum motion.
    """
    # Parameters
    L = 1.0  # Length in meters
    g = 9.81  # Gravitational acceleration in m/s²
    omega_0_sq = g / L  # Natural frequency squared
    omega_0 = np.sqrt(omega_0_sq)  # Natural frequency
    
    # Time settings
    t_span = (0, 20)  # Time span
    t_eval = np.linspace(*t_span, 1000)  # Time points for evaluation
    
    # Initial conditions
    y0 = [np.radians(15), 0]  # Initial angle (15 degrees) and angular velocity (0)
    
    # Different damping coefficients
    damping_values = [0.05, 0.2, 0.5, 1.0, 2.0]  # From underdamped to overdamped
    colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A8', '#33A8FF']
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Solve and plot for each damping coefficient
    for b, color in zip(damping_values, colors):
        # No driving force for this example
        A = 0
        omega_d = 0
        
        # Solve the ODE
        sol = solve_ivp(
            lambda t, y: pendulum_system(t, y, b, A, omega_d, omega_0_sq),
            t_span, y0, t_eval=t_eval, method='RK45'
        )
        
        # Plot the angular position vs time
        plt.plot(sol.t, np.degrees(sol.y[0]), color=color, linewidth=2,
                 label=f'b = {b:.2f} ({"underdamped" if b < 2*omega_0 else "critically damped" if b == 2*omega_0 else "overdamped"})'
                )
    
    # Add critical damping line
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Add grid, labels, and title
    plt.grid(True, alpha=0.3)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Angular Position (degrees)', fontsize=12)
    plt.title('Effect of Damping Coefficient on Pendulum Motion', fontsize=14)
    plt.legend()
    
    # Add text box with parameters
    textstr = f'Length: {L} m\nNatural frequency: {omega_0:.2f} rad/s\nCritical damping: {2*omega_0:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'damping_effect.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_resonance_curve():
    """
    Generate a resonance curve showing amplitude vs driving frequency.
    """
    # Parameters
    L = 1.0  # Length in meters
    g = 9.81  # Gravitational acceleration in m/s²
    omega_0_sq = g / L  # Natural frequency squared
    omega_0 = np.sqrt(omega_0_sq)  # Natural frequency
    
    # Driving amplitude
    A = 0.5
    
    # Different damping coefficients
    damping_values = [0.1, 0.3, 0.5, 1.0]
    colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A8']
    
    # Range of driving frequencies
    omega_d_range = np.linspace(0.1, 2.5 * omega_0, 100)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Calculate and plot amplitude vs driving frequency for each damping coefficient
    for b, color in zip(damping_values, colors):
        # Calculate amplitude using the steady-state solution formula
        amplitudes = []
        for omega_d in omega_d_range:
            amplitude = A / np.sqrt((omega_0_sq - omega_d**2)**2 + (b * omega_d)**2)
            amplitudes.append(amplitude)
        
        # Plot the amplitude vs driving frequency
        plt.plot(omega_d_range / omega_0, amplitudes, color=color, linewidth=2,
                 label=f'b = {b:.1f}')
    
    # Mark the natural frequency
    plt.axvline(x=1, color='black', linestyle='--', alpha=0.5, label='Natural frequency')
    
    # Add grid, labels, and title
    plt.grid(True, alpha=0.3)
    plt.xlabel('Driving Frequency / Natural Frequency (ω_d/ω_0)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.title('Resonance Curve: Amplitude vs Driving Frequency', fontsize=14)
    plt.legend()
    
    # Add text box with formula
    textstr = 'Amplitude = $\\frac{A}{\\sqrt{(\\omega_0^2 - \\omega_d^2)^2 + (b\\omega_d)^2}}$'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'resonance_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_phase_space():
    """
    Generate a phase space plot (angular position vs angular velocity).
    """
    # Parameters
    L = 1.0  # Length in meters
    g = 9.81  # Gravitational acceleration in m/s²
    omega_0_sq = g / L  # Natural frequency squared
    omega_0 = np.sqrt(omega_0_sq)  # Natural frequency
    
    # Time settings
    t_span = (0, 100)  # Long time span to reach steady state
    t_eval = np.linspace(*t_span, 10000)  # Many time points for smooth trajectory
    
    # Initial conditions
    y0 = [np.radians(15), 0]  # Initial angle (15 degrees) and angular velocity (0)
    
    # Parameters for chaotic motion
    b = 0.2  # Low damping
    A = 1.5  # High driving amplitude
    omega_d = 2/3 * omega_0  # Driving frequency as a fraction of natural frequency
    
    # Solve the ODE
    sol = solve_ivp(
        lambda t, y: pendulum_system(t, y, b, A, omega_d, omega_0_sq),
        t_span, y0, t_eval=t_eval, method='RK45'
    )
    
    # Extract position and velocity, and wrap position to [-π, π]
    theta = np.remainder(sol.y[0] + np.pi, 2 * np.pi) - np.pi
    omega = sol.y[1]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot phase space trajectory with color gradient by time
    points = plt.scatter(theta, omega, c=sol.t, cmap='viridis', s=1, alpha=0.5)
    plt.colorbar(points, label='Time (s)')
    
    # Add grid, labels, and title
    plt.grid(True, alpha=0.3)
    plt.xlabel('Angular Position (radians)', fontsize=12)
    plt.ylabel('Angular Velocity (radians/s)', fontsize=12)
    plt.title('Phase Space Trajectory of Forced Damped Pendulum', fontsize=14)
    
    # Add text box with parameters
    textstr = f'Damping: b = {b}\nDriving amplitude: A = {A}\nDriving frequency: ω_d = {omega_d:.2f} rad/s\n(ω_d/ω_0 = {omega_d/omega_0:.2f})'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'phase_space.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_poincare_section():
    """
    Generate a Poincaré section for the forced damped pendulum.
    """
    # Parameters
    L = 1.0  # Length in meters
    g = 9.81  # Gravitational acceleration in m/s²
    omega_0_sq = g / L  # Natural frequency squared
    omega_0 = np.sqrt(omega_0_sq)  # Natural frequency
    
    # Time settings for a long simulation
    t_span = (0, 500)  # Very long time span
    t_step = 0.01
    t_eval = np.arange(t_span[0], t_span[1], t_step)  # Many time points
    
    # Initial conditions
    y0 = [np.radians(15), 0]  # Initial angle (15 degrees) and angular velocity (0)
    
    # Parameters for chaotic motion
    b = 0.2  # Low damping
    A = 1.5  # High driving amplitude
    omega_d = 2/3 * omega_0  # Driving frequency
    
    # Solve the ODE
    sol = solve_ivp(
        lambda t, y: pendulum_system(t, y, b, A, omega_d, omega_0_sq),
        t_span, y0, t_eval=t_eval, method='RK45'
    )
    
    # Extract position and velocity, and wrap position to [-π, π]
    theta = np.remainder(sol.y[0] + np.pi, 2 * np.pi) - np.pi
    omega = sol.y[1]
    
    # Create Poincaré section by sampling at driving period
    T_d = 2 * np.pi / omega_d  # Driving period
    indices = []
    
    # Find indices where time is approximately a multiple of the driving period
    for i in range(len(sol.t)):
        if i > 0 and (sol.t[i] % T_d) < t_step:
            indices.append(i)
    
    # Extract points for the Poincaré section
    theta_poincare = theta[indices]
    omega_poincare = omega[indices]
    
    # Discard transient behavior (first 20% of points)
    discard = int(0.2 * len(indices))
    theta_poincare = theta_poincare[discard:]
    omega_poincare = omega_poincare[discard:]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot Poincaré section
    plt.scatter(theta_poincare, omega_poincare, s=5, color='blue', alpha=0.7)
    
    # Add grid, labels, and title
    plt.grid(True, alpha=0.3)
    plt.xlabel('Angular Position (radians)', fontsize=12)
    plt.ylabel('Angular Velocity (radians/s)', fontsize=12)
    plt.title('Poincaré Section of Forced Damped Pendulum', fontsize=14)
    
    # Add text box with parameters
    textstr = f'Damping: b = {b}\nDriving amplitude: A = {A}\nDriving frequency: ω_d = {omega_d:.2f} rad/s\n(ω_d/ω_0 = {omega_d/omega_0:.2f})'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'poincare_section.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_bifurcation_diagram():
    """
    Generate a bifurcation diagram for the forced damped pendulum.
    """
    # Parameters
    L = 1.0  # Length in meters
    g = 9.81  # Gravitational acceleration in m/s²
    omega_0_sq = g / L  # Natural frequency squared
    omega_0 = np.sqrt(omega_0_sq)  # Natural frequency
    
    # Fixed parameters
    b = 0.2  # Damping coefficient
    omega_d = 2/3 * omega_0  # Driving frequency
    
    # Range of driving amplitudes
    A_values = np.linspace(0.1, 2.0, 30)
    
    # Time settings
    t_span = (0, 200)  # Long time span
    t_step = 0.01
    t_eval = np.arange(t_span[0], t_span[1], t_step)
    
    # Initial conditions
    y0 = [np.radians(15), 0]  # Initial angle and angular velocity
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # For each driving amplitude
    for A in A_values:
        # Solve the ODE
        sol = solve_ivp(
            lambda t, y: pendulum_system(t, y, b, A, omega_d, omega_0_sq),
            t_span, y0, t_eval=t_eval, method='RK45'
        )
        
        # Extract position and wrap to [-π, π]
        theta = np.remainder(sol.y[0] + np.pi, 2 * np.pi) - np.pi
        
        # Create Poincaré section
        T_d = 2 * np.pi / omega_d  # Driving period
        indices = []
        
        # Find indices where time is approximately a multiple of the driving period
        for i in range(len(sol.t)):
            if i > 0 and (sol.t[i] % T_d) < t_step:
                indices.append(i)
        
        # Extract points for the Poincaré section
        theta_poincare = theta[indices]
        
        # Discard transient behavior (first 50% of points)
        discard = int(0.5 * len(indices))
        theta_poincare = theta_poincare[discard:]
        
        # Plot points in the bifurcation diagram
        plt.plot(np.ones_like(theta_poincare) * A, theta_poincare, 'k.', markersize=0.5, alpha=0.5)
    
    # Add grid, labels, and title
    plt.grid(True, alpha=0.3)
    plt.xlabel('Driving Amplitude (A)', fontsize=12)
    plt.ylabel('Angular Position at Poincaré Section (radians)', fontsize=12)
    plt.title('Bifurcation Diagram of Forced Damped Pendulum', fontsize=14)
    
    # Add text box with parameters
    textstr = f'Damping: b = {b}\nDriving frequency: ω_d = {omega_d:.2f} rad/s\n(ω_d/ω_0 = {omega_d/omega_0:.2f})'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'bifurcation_diagram.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Generating forced damped pendulum visualizations...")
    
    # Generate all visualizations
    generate_damping_effect()
    print("Created damping effect plot")
    
    generate_resonance_curve()
    print("Created resonance curve plot")
    
    generate_phase_space()
    print("Created phase space plot")
    
    generate_poincare_section()
    print("Created Poincaré section plot")
    
    generate_bifurcation_diagram()
    print("Created bifurcation diagram")
    
    print(f"\nAll visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
