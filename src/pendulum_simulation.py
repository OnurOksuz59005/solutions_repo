import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import os

# Create directory for images if it doesn't exist
image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs', '1 Physics', '1 Mechanics', 'images')
os.makedirs(image_dir, exist_ok=True)

# Function to solve the forced damped pendulum equation
def pendulum_system(t, y, b, A, omega_d, omega_0_sq):
    """System of first-order ODEs for the forced damped pendulum.
    
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

# Function to solve the linearized forced damped pendulum equation
def linear_pendulum_system(t, y, b, A, omega_d, omega_0_sq):
    """System of first-order ODEs for the linearized forced damped pendulum.
    
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

# Parameters
g = 9.81  # Gravitational acceleration (m/s²)
L = 1.0   # Pendulum length (m)
omega_0_sq = g / L  # Natural frequency squared
omega_0 = np.sqrt(omega_0_sq)  # Natural frequency

# Time settings
t_span = (0, 50)  # Time span
t_eval = np.linspace(*t_span, 5000)  # Time points for evaluation

# Initial conditions
y0 = [0.2, 0]  # [theta_0, omega_0]

# 1. Effect of damping coefficient
damping_coefficients = [0.1, 0.5, 1.0, 2.0]
A = 0.5  # Fixed driving amplitude
omega_d = omega_0  # Fixed driving frequency at resonance

plt.figure(figsize=(12, 8))

for i, b in enumerate(damping_coefficients):
    # Solve the ODE
    sol = solve_ivp(
        lambda t, y: pendulum_system(t, y, b, A, omega_d, omega_0_sq),
        t_span, y0, t_eval=t_eval, method='RK45'
    )
    
    # Plot the solution
    plt.subplot(2, 2, i+1)
    plt.plot(sol.t, sol.y[0], 'b-')
    plt.grid(True)
    plt.title(f'Damping Coefficient b = {b}')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')

plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'damping_effect.png'), dpi=300, bbox_inches='tight')

# 2. Effect of driving amplitude
driving_amplitudes = [0.1, 0.5, 1.0, 2.0]
b = 0.5  # Fixed damping coefficient

plt.figure(figsize=(12, 8))

for i, A in enumerate(driving_amplitudes):
    # Solve the ODE
    sol = solve_ivp(
        lambda t, y: pendulum_system(t, y, b, A, omega_d, omega_0_sq),
        t_span, y0, t_eval=t_eval, method='RK45'
    )
    
    # Plot the solution
    plt.subplot(2, 2, i+1)
    plt.plot(sol.t, sol.y[0], 'r-')
    plt.grid(True)
    plt.title(f'Driving Amplitude A = {A}')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')

plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'amplitude_effect.png'), dpi=300, bbox_inches='tight')

# 3. Resonance curve
driving_frequencies = np.linspace(0.5 * omega_0, 1.5 * omega_0, 100)
b = 0.2  # Fixed damping coefficient
A = 0.5  # Fixed driving amplitude

# Time settings for steady state
t_span_steady = (0, 100)  # Longer time to reach steady state
t_eval_steady = np.linspace(80, 100, 1000)  # Only evaluate the last 20 seconds

# Calculate amplitude for each driving frequency
amplitudes = []

for omega_d in driving_frequencies:
    # Solve the ODE
    sol = solve_ivp(
        lambda t, y: pendulum_system(t, y, b, A, omega_d, omega_0_sq),
        t_span_steady, y0, t_eval=t_eval_steady, method='RK45'
    )
    
    # Calculate the amplitude (max - min) / 2
    amplitude = (np.max(sol.y[0]) - np.min(sol.y[0])) / 2
    amplitudes.append(amplitude)

# Plot the resonance curve
plt.figure(figsize=(10, 6))
plt.plot(driving_frequencies / omega_0, amplitudes, 'g-', linewidth=2)
plt.grid(True)
plt.axvline(x=1.0, color='r', linestyle='--', label='Natural Frequency')
plt.xlabel('Driving Frequency / Natural Frequency')
plt.ylabel('Amplitude (rad)')
plt.title('Resonance Curve for Forced Damped Pendulum')
plt.legend()
plt.savefig(os.path.join(image_dir, 'resonance_curve.png'), dpi=300, bbox_inches='tight')

# 4. Phase space trajectories
b = 0.1  # Low damping
A = 1.2  # High driving amplitude
omega_d = 2/3 * omega_0  # Non-resonant driving frequency

# Solve the ODE for a longer time to observe complex behavior
t_span_long = (0, 200)
t_eval_long = np.linspace(*t_span_long, 10000)

sol_chaotic = solve_ivp(
    lambda t, y: pendulum_system(t, y, b, A, omega_d, omega_0_sq),
    t_span_long, y0, t_eval=t_eval_long, method='RK45'
)

# Plot the phase space trajectory
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(sol_chaotic.t, sol_chaotic.y[0], 'b-')
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.title('Angle vs Time (Potentially Chaotic Regime)')

plt.subplot(2, 1, 2)
plt.plot(sol_chaotic.y[0], sol_chaotic.y[1], 'r-')
plt.grid(True)
plt.xlabel('Angle (rad)')
plt.ylabel('Angular Velocity (rad/s)')
plt.title('Phase Space Trajectory')
plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'phase_space.png'), dpi=300, bbox_inches='tight')

# 5. Poincaré section
# Extract points at driving period intervals
driving_period = 2 * np.pi / omega_d
n_periods = int(t_span_long[1] / driving_period)
poincare_points = []

for i in range(n_periods):
    t_target = i * driving_period
    # Find the closest time point
    idx = np.argmin(np.abs(sol_chaotic.t - t_target))
    if idx < len(sol_chaotic.t):
        poincare_points.append([sol_chaotic.y[0][idx], sol_chaotic.y[1][idx]])

poincare_points = np.array(poincare_points)

# Plot the Poincaré section
plt.figure(figsize=(8, 8))
plt.scatter(poincare_points[:, 0], poincare_points[:, 1], s=5, c='blue')
plt.grid(True)
plt.xlabel('Angle (rad)')
plt.ylabel('Angular Velocity (rad/s)')
plt.title('Poincaré Section (Sampled at Driving Period)')
plt.axis('equal')
plt.savefig(os.path.join(image_dir, 'poincare_section.png'), dpi=300, bbox_inches='tight')

# 6. Bifurcation diagram
# Varying the driving amplitude
A_values = np.linspace(0.1, 1.5, 100)
b = 0.1  # Fixed damping
omega_d = 2/3 * omega_0  # Fixed driving frequency

# Time settings for steady state
t_span_bif = (0, 200)
t_eval_bif = np.linspace(100, 200, 5000)  # Only evaluate the last 100 seconds

bifurcation_data = []

for A_val in A_values:
    # Solve the ODE
    sol = solve_ivp(
        lambda t, y: pendulum_system(t, y, b, A_val, omega_d, omega_0_sq),
        t_span_bif, y0, t_eval=t_eval_bif, method='RK45'
    )
    
    # Extract points at driving period intervals
    driving_period = 2 * np.pi / omega_d
    t_mod = sol.t % driving_period
    indices = np.where(t_mod < driving_period/50)[0]  # Points close to multiples of the driving period
    
    # Store the angle values at these points along with the corresponding A value
    for idx in indices:
        bifurcation_data.append([A_val, sol.y[0][idx]])

bifurcation_data = np.array(bifurcation_data)

# Plot the bifurcation diagram
plt.figure(figsize=(10, 6))
plt.scatter(bifurcation_data[:, 0], bifurcation_data[:, 1], s=0.5, c='black')
plt.grid(True)
plt.xlabel('Driving Amplitude (A)')
plt.ylabel('Angle (rad)')
plt.title('Bifurcation Diagram (Varying Driving Amplitude)')
plt.savefig(os.path.join(image_dir, 'bifurcation_diagram.png'), dpi=300, bbox_inches='tight')

# 7. Compare linear and nonlinear solutions
b = 0.2  # Moderate damping
A = 0.5  # Moderate driving amplitude
omega_d = omega_0  # Resonant driving frequency

# Solve both linear and nonlinear equations
sol_linear = solve_ivp(
    lambda t, y: linear_pendulum_system(t, y, b, A, omega_d, omega_0_sq),
    t_span, y0, t_eval=t_eval, method='RK45'
)

sol_nonlinear = solve_ivp(
    lambda t, y: pendulum_system(t, y, b, A, omega_d, omega_0_sq),
    t_span, y0, t_eval=t_eval, method='RK45'
)

# Plot the comparison
plt.figure(figsize=(12, 6))
plt.plot(sol_linear.t, sol_linear.y[0], 'b-', label='Linear Approximation')
plt.plot(sol_nonlinear.t, sol_nonlinear.y[0], 'r--', label='Nonlinear Solution')
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.title('Comparison of Linear and Nonlinear Solutions')
plt.legend()
plt.savefig(os.path.join(image_dir, 'linear_vs_nonlinear.png'), dpi=300, bbox_inches='tight')

print(f"Images saved to {image_dir}")
