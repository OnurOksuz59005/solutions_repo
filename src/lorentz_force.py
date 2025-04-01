import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Create directory for images if it doesn't exist
image_dir = os.path.join('docs', '1 Physics', '4 Electromagnetism', 'images')
os.makedirs(image_dir, exist_ok=True)

# Physical constants
ELECTRON_CHARGE = -1.602e-19  # Coulombs
ELECTRON_MASS = 9.109e-31     # kg
PROTON_CHARGE = 1.602e-19     # Coulombs
PROTON_MASS = 1.673e-27       # kg

# Numerical integration methods
def runge_kutta_step(q, m, E, v, r, B, dt):
    """
    Perform a single step of 4th-order Runge-Kutta integration for a charged particle
    in electromagnetic fields.
    
    Args:
        q (float): Charge of the particle
        m (float): Mass of the particle
        E (numpy.ndarray): Electric field vector
        v (numpy.ndarray): Current velocity vector
        r (numpy.ndarray): Current position vector
        B (numpy.ndarray): Magnetic field vector
        dt (float): Time step
        
    Returns:
        tuple: New position and velocity vectors
    """
    # Define the derivatives function for position and velocity
    def derivatives(v_in, r_in):
        # Lorentz force: F = q(E + v × B)
        # a = F/m = (q/m)(E + v × B)
        a = (q/m) * (E + np.cross(v_in, B))
        return v_in, a
    
    # RK4 algorithm
    v1, a1 = derivatives(v, r)
    v2, a2 = derivatives(v + 0.5*dt*a1, r + 0.5*dt*v1)
    v3, a3 = derivatives(v + 0.5*dt*a2, r + 0.5*dt*v2)
    v4, a4 = derivatives(v + dt*a3, r + dt*v3)
    
    # Update position and velocity
    r_new = r + (dt/6) * (v1 + 2*v2 + 2*v3 + v4)
    v_new = v + (dt/6) * (a1 + 2*a2 + 2*a3 + a4)
    
    return r_new, v_new

def simulate_particle_motion(q, m, E, B, v0, r0, dt, steps):
    """
    Simulates particle motion using Runge-Kutta 4th order method.
    Handles constant or position-dependent B fields.
    
    Args:
        q (float): Charge
        m (float): Mass
        E (np.array or callable): Electric field vector or function E(r)
        B (np.array or callable): Magnetic field vector or function B(r)
        v0 (np.array): Initial velocity
        r0 (np.array): Initial position
        dt (float): Time step
        steps (int): Number of steps
        
    Returns:
        tuple: (positions, velocities) arrays
    """
    positions = np.zeros((steps, 3))
    velocities = np.zeros((steps, 3))
    
    positions[0] = r0
    velocities[0] = v0
    
    # Check if fields are functions or constant vectors
    E_is_func = callable(E)
    B_is_func = callable(B)
    
    for i in range(1, steps):
        # Get fields at the current position (r_{i-1})
        current_r = positions[i-1]
        current_E = E(current_r) if E_is_func else E
        current_B = B(current_r) if B_is_func else B
        
        # Perform RK4 step with the fields at the current position
        try:
            positions[i], velocities[i] = runge_kutta_step(
                q, m, current_E, velocities[i-1], current_r, current_B, dt
            )
        except Exception as e:
            print(f"Error during RK4 step {i}: {e}")
            print(f"  Position: {current_r}")
            print(f"  Velocity: {velocities[i-1]}")
            print(f"  E-Field: {current_E}")
            print(f"  B-Field: {current_B}")
            # Truncate results and return what we have
            return positions[:i], velocities[:i]

        # Optional: Check for runaway values
        if np.any(np.isnan(positions[i])) or np.any(np.isinf(positions[i])):
            print(f"Warning: NaN or Inf detected at step {i}. Stopping simulation.")
            return positions[:i], velocities[:i]
            
    return positions, velocities

# Visualization functions
def plot_trajectory_3d(positions, title, save_path=None, E=None, B=None, is_magnetic_bottle=False):
    """
    Plot the 3D trajectory of a particle.
    
    Args:
        positions (numpy.ndarray): Array of particle positions
        title (str): Plot title
        save_path (str, optional): Path to save the plot image
        E (numpy.ndarray, optional): Electric field vector
        B (numpy.ndarray, optional): Magnetic field vector
        is_magnetic_bottle (bool, optional): Whether this is a magnetic bottle simulation
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'g-', linewidth=1.5, label='Trajectory')
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
    
    # For magnetic bottle, plot B field as a straight line
    if is_magnetic_bottle:
        # Plot a straight blue line along the z-axis to represent the B field
        ax.plot([0, 0], [0, 0], [z_min, z_max], 'b-', linewidth=2, label='B field')
        # Add 'B' label at the top of the line
        ax.text(0, 0, z_max*1.05, 'B', color='blue', fontsize=14, ha='center')
    elif B is not None and np.any(B):
        # For uniform fields, plot magnetic field vector in a corner
        corner_x = x_min + 0.1 * (x_max - x_min)
        corner_y = y_max - 0.1 * (y_max - y_min)
        corner_z = z_max - 0.1 * (z_max - z_min)
        
        B_scale = 0.15 * max_range
        ax.quiver(corner_x, corner_y, corner_z, 
                 B_scale * B[0], B_scale * B[1], B_scale * B[2], 
                 color='b', arrow_length_ratio=0.3, label='B field')
    
    # Add electric field vector if provided
    if E is not None and np.any(E):
        # Plot electric field vector in top corner
        corner_x = x_max - 0.1 * (x_max - x_min)
        corner_y = y_max - 0.1 * (y_max - y_min)
        corner_z = z_max - 0.1 * (z_max - z_min)
        
        E_scale = 0.15 * max_range
        ax.quiver(corner_x, corner_y, corner_z, 
                 E_scale * E[0], E_scale * E[1], E_scale * E[2], 
                 color='r', arrow_length_ratio=0.3, label='E field')
    
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
    
    plt.close(fig)

def create_magnetic_bottle_static_image(positions, title="Charged Particle in Magnetic Bottle (Static View)", save_path=None):
    """
    Create a static image of the magnetic bottle from a different perspective.
    
    Args:
        positions (numpy.ndarray): Array of particle positions
        title (str): Plot title
        save_path (str, optional): Path to save the plot image
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'r-', linewidth=1.5, label='Trajectory')
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='g', s=100, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='r', s=100, label='End')
    
    # Calculate axis limits
    z_min = np.min(positions[:, 2])
    z_max = np.max(positions[:, 2])
    
    # Extend the z-range by 20% for better visualization
    z_range = z_max - z_min
    z_min -= 0.2 * z_range
    z_max += 0.2 * z_range
    
    # Plot a straight blue line along the z-axis to represent the B field
    ax.plot([0, 0], [0, 0], [z_min, z_max], 'b-', linewidth=2, label='B field')
    
    # Add 'B' label at the top of the line
    ax.text(0, 0, z_max*1.05, 'B', color='blue', fontsize=14, ha='center')
    
    # Set a different viewing angle for the static image
    ax.view_init(elev=20, azim=70)  # Different perspective from the default
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend(loc='upper left')
    
    # Format tick labels
    ax.ticklabel_format(style='sci', scilimits=(-2, 2), axis='both')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved static image to {save_path}")
    
    plt.close(fig)

# Simulation scenarios
def uniform_magnetic_field(q=ELECTRON_CHARGE, m=ELECTRON_MASS, B0=1.0, v_perp=1e6, v_parallel=0.0, steps=None, dt=None, save_plots=True):
    """
    Simulate a charged particle in a uniform magnetic field.
    Allows for both pure circular (v_parallel=0) and helical motion (v_parallel != 0).

    Args:
        q (float): Charge of the particle
        m (float): Mass of the particle
        B0 (float): Magnetic field strength (along z-axis)
        v_perp (float): Initial velocity component perpendicular to B (in x-direction)
        v_parallel (float): Initial velocity component parallel to B (in z-direction)
        steps (int, optional): Number of simulation steps. If None, calculated for 3 orbits.
        dt (float, optional): Time step. If None, calculated as T_cyclotron / 100.
        save_plots (bool): Whether to save plots

    Returns:
        tuple: Arrays of positions and velocities, and observations dict.
    """
    # Set up fields and initial conditions
    E = np.array([0.0, 0.0, 0.0])
    B = np.array([0.0, 0.0, B0])
    v0_vec = np.array([v_perp, 0.0, v_parallel]) # Initial velocity
    r0 = np.array([0.0, 0.0, 0.0])

    # --- Parameter Calculation --- 
    # Theoretical values
    if B0 == 0: 
        print("Warning: B0 is zero, cannot calculate cyclotron motion.")
        return np.array([r0]), np.array([v0_vec]), {}
    
    cyclotron_freq_mag = np.abs(q * B0 / m)
    if cyclotron_freq_mag == 0:
        print("Warning: Cyclotron frequency is zero.")
        # Particle moves in a straight line if B=0 or q=0 or m=inf
        # Handle this case if necessary, for now return basic info
        return np.array([r0]), np.array([v0_vec]), {}
        
    larmor_radius = m * v_perp / (np.abs(q) * B0) if v_perp != 0 else 0
    period = 2 * np.pi / cyclotron_freq_mag

    # Determine dt and steps if not provided
    if dt is None:
        dt = period / 100.0 # Time step as fraction of period
    if steps is None:
        num_orbits = 3
        steps = int(num_orbits * period / dt)
    
    print(f"Simulation Parameters: dt={dt:.2e} s, steps={steps}, T_cyc={period:.2e} s")

    # --- Simulation --- 
    positions, velocities = simulate_particle_motion(q, m, E, B, v0_vec, r0, dt, steps)

    # --- Measurement --- 
    # Measured Radius: Max distance from the guiding center (z-axis for v_parallel=0)
    # For helical motion, max distance from the axis of the helix
    # PREVIOUSLY: xy_distances = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
    # PREVIOUSLY: measured_radius = np.max(xy_distances) if len(xy_distances) > 0 else 0
    # CORRECTED: For v0=(v_perp, 0, 0) starting at origin, the max x-excursion is the radius.
    measured_radius = np.max(np.abs(positions[:, 0])) if positions.shape[0] > 0 else 0

    # Measured Period/Frequency: Find time to return to similar state (pos & vel)
    measured_period = None
    # Find first time x-velocity is positive again after half a period (approx)
    half_period_steps = int(0.5 * period / dt)
    quarter_period_steps = int(0.25 * period / dt)

    if steps > half_period_steps + 10 and quarter_period_steps > 0:
        # Look for the point near one full period later
        target_step = int(period / dt) 
        search_range = int(0.2 * period / dt) # Search +/- 10% around expected period
        min_diff = float('inf')
        best_step = -1

        for i in range(max(1, target_step - search_range), min(steps, target_step + search_range)):
            # Compare state at step i with initial state (step 0)
            # More robust: compare angle in xy plane
            angle_i = np.arctan2(positions[i, 1], positions[i, 0])
            angle_0 = np.arctan2(positions[0, 1], positions[0, 0]) # Should be 0 here
            angle_diff = np.abs(angle_i - angle_0)
            # Check if angle is close to 0 or 2*pi (handle wrap around if needed)
            if angle_diff < 0.1 or np.abs(angle_diff - 2*np.pi) < 0.1:
                # Check if z position is also similar (for helical)
                z_diff = np.abs(positions[i, 2] - positions[0, 2] - v_parallel * (i * dt))
                if z_diff < larmor_radius * 0.1: # Allow some tolerance in z
                    diff = angle_diff + z_diff # Combine errors
                    if diff < min_diff:
                         min_diff = diff
                         best_step = i

        if best_step > 0:
            measured_period = best_step * dt

    if measured_period is None or measured_period <= 0:
        measured_period = period # Fallback to theoretical
        measured_freq = cyclotron_freq_mag
        print("Note: Using theoretical period for measurement.")
    else:
        measured_freq = 2 * np.pi / measured_period

    # --- Observations --- 
    observations = {
        'theoretical_radius': f"Larmor radius (theory): {larmor_radius:.3e} m",
        'measured_radius': f"Larmor radius (measured): {measured_radius:.3e} m",
        'theoretical_freq': f"Cyclotron frequency (theory): {cyclotron_freq_mag:.3e} rad/s",
        'measured_freq': f"Cyclotron frequency (measured): {measured_freq:.3e} rad/s",
        'error_radius': f"Error in radius: {100 * abs(measured_radius - larmor_radius) / larmor_radius if larmor_radius != 0 else 0:.2f}%",
        'error_freq': f"Error in frequency: {100 * abs(measured_freq - cyclotron_freq_mag) / cyclotron_freq_mag if cyclotron_freq_mag != 0 else 0:.2f}%"
    }
    for key, value in observations.items():
        print(value)

    # --- Plotting --- 
    if save_plots:
        # Common plot settings
        line_color = 'red'
        b_field_color = 'blue'
        start_marker = 'go' # Green circle for start
        end_marker = 'ro' # Red circle for end

        # --- 2D Plot (XY Projection) --- 
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.plot(positions[:, 0], positions[:, 1], color=line_color, linewidth=1.5, label='Particle Path (XY Projection)')
        ax2.plot(positions[0, 0], positions[0, 1], start_marker, markersize=8, label='Start')
        ax2.plot(positions[-1, 0], positions[-1, 1], end_marker, markersize=8, label='End')
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        title_2d = "Cyclotron Motion (XY Projection)" if v_parallel == 0 else "Helical Motion (XY Projection)"
        ax2.set_title(title_2d)
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.set_aspect('equal', adjustable='box')
        ax2.ticklabel_format(style='sci', scilimits=(-2,2), axis='both')
        # Center the plot around the circle/projection
        center_x = np.mean(positions[:, 0])
        center_y = np.mean(positions[:, 1])
        max_dev = max(np.max(np.abs(positions[:, 0] - center_x)), np.max(np.abs(positions[:, 1] - center_y))) * 1.2
        ax2.set_xlim(center_x - max_dev, center_x + max_dev)
        ax2.set_ylim(center_y - max_dev, center_y + max_dev)

        plot_path_2d = os.path.join(image_dir, 'uniform_magnetic_field_2d.png')
        plt.savefig(plot_path_2d, dpi=300, bbox_inches='tight')
        print(f"Saved 2D plot to {plot_path_2d}")
        plt.close(fig2)

        # --- 3D Plot --- 
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color=line_color, linewidth=1.5, label='Particle Path')
        ax.plot(positions[0, 0], positions[0, 1], positions[0, 2], start_marker, markersize=8, label='Start')
        ax.plot(positions[-1, 0], positions[-1, 1], positions[-1, 2], end_marker, markersize=8, label='End')

        # Add B field vector
        # Scale length relative to trajectory size for better visibility
        max_dim = max(np.ptp(positions[:,0]), np.ptp(positions[:,1]), np.ptp(positions[:,2])) if positions.shape[0] > 1 else 1.0
        b_arrow_length = max_dim * 0.3
        ax.quiver(0, 0, 0, 0, 0, b_arrow_length, color=b_field_color, arrow_length_ratio=0.2, label='B field')

        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        title_3d = "Cyclotron Motion (3D View)" if v_parallel == 0 else "Helical Motion (3D View)"
        ax.set_title(title_3d)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.ticklabel_format(style='sci', axis='both', scilimits=(-2,2))
        # Try to make aspect ratio somewhat equal if possible, but prioritize showing the path
        try:
            ptp_x = np.ptp(positions[:,0])
            ptp_y = np.ptp(positions[:,1])
            ptp_z = np.ptp(positions[:,2])
            # Only set aspect if dimensions are non-zero to avoid singular matrix
            if ptp_x > 1e-9 and ptp_y > 1e-9 and ptp_z > 1e-9:
                 ax.set_box_aspect([ptp_x, ptp_y, ptp_z])
            else:
                 # Default aspect for planar or linear data
                 ax.set_aspect('equal', adjustable='box') # Might be better for circular path
        except Exception as e:
            print(f"Warning: Could not set 3D aspect ratio: {e}")
            pass # Fallback if data is flat or other issues occur

        plot_path = os.path.join(image_dir, 'uniform_magnetic_field.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved 3D plot to {plot_path}")
        plt.close(fig)

    return positions, velocities, observations

def crossed_fields(q=ELECTRON_CHARGE, m=ELECTRON_MASS, E0=1e5, B0=1.0, num_periods=5, steps_per_period=50, save_plots=True):
    """
    Simulate a charged particle in crossed electric and magnetic fields (E⊥B).
    E is along y, B is along z. Drift should be along x.
    Calculates dt based on cyclotron period for better resolution.
    """
    # Calculate cyclotron period and dt
    if abs(q) > 1e-20 and abs(B0) > 1e-9:
        omega_c = abs(q * B0 / m)
        T_cyc = 2 * np.pi / omega_c
        dt = T_cyc / steps_per_period
        steps = int(num_periods * steps_per_period)
    else: # Fallback if B0 or q is zero/tiny
        dt = 1e-11 
        steps = 1000
        T_cyc = np.inf
        print("Warning: B0 or q is near zero, using default dt/steps for crossed fields.")

    # Setup fields and initial conditions
    E = np.array([0.0, E0, 0.0]) 
    B = np.array([0.0, 0.0, B0]) 
    v0 = np.array([0.0, 0.0, 0.0])
    r0 = np.array([0.0, 0.0, 0.0])
    
    theoretical_drift_v = E0 / B0 if B0 != 0 else 0
    
    print(f"Simulation Parameters: dt={dt:.3e} s, steps={steps}, T_cyc={T_cyc:.3e} s")
    positions, velocities = simulate_particle_motion(q, m, E, B, v0, r0, dt, steps)
    
    # Measured drift velocity (more robustly)
    if steps > steps_per_period: # Use average over last few periods
        start_avg_idx = max(0, steps - 3 * steps_per_period) # Avg over last 3 periods
        avg_vx = np.mean(velocities[start_avg_idx:, 0])
    elif steps > 10:
        avg_vx = np.mean(velocities[steps//2:, 0]) # Use last half
    else:
        avg_vx = np.mean(velocities[:, 0]) if steps > 0 else 0
    
    measured_drift_v = avg_vx
    drift_error = 100 * abs(measured_drift_v - theoretical_drift_v) / theoretical_drift_v if theoretical_drift_v != 0 else 0

    observations = {
        'theoretical_drift': f"E×B drift velocity (theory): {theoretical_drift_v:.3e} m/s",
        'measured_drift': f"E×B drift velocity (measured): {measured_drift_v:.3e} m/s",
        'error_drift': f"Error in drift velocity: {drift_error:.2f}%"
    }
    for key, value in observations.items():
        print(value)

    if save_plots:
        # === 3D Plot (Optional - keep for context if needed) ===
        # ... (consider removing or simplifying if only 2D is needed)
        # === Keep the saving part for 3D ===
        fig_3d = plt.figure(figsize=(10, 8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], color='red', linewidth=1.0, label='Particle Path')
        # ... rest of 3D plot code ...
        max_range_xy = max(np.ptp(positions[:,0]), np.ptp(positions[:,1])) if positions.shape[0] > 1 else 1.0
        vec_length = max_range_xy * 0.25 
        ax_3d.quiver(0, 0, 0, 0, vec_length * (E0/abs(E0)) if E0 != 0 else 0, 0, color='green', arrow_length_ratio=0.3, label='E field (+y)')
        ax_3d.quiver(0, 0, 0, 0, 0, vec_length * (B0/abs(B0)) if B0 != 0 else 0, color='blue', arrow_length_ratio=0.3, label='B field (+z)')
        ax_3d.set_xlabel('X (m) Drift'); ax_3d.set_ylabel('Y (m)'); ax_3d.set_zlabel('Z (m)')
        ax_3d.set_title('Crossed E×B Drift (3D View)')
        ax_3d.legend(); ax_3d.grid(True, linestyle='--', alpha=0.3)
        ax_3d.ticklabel_format(style='sci', axis='both', scilimits=(-3,3)); ax_3d.view_init(elev=30, azim=-60)
        try:
            ptp_x = np.ptp(positions[:,0]); ptp_y = np.ptp(positions[:,1]); ptp_z = np.ptp(positions[:,2])
            if ptp_z < 1e-9: ax_3d.set_box_aspect([ptp_x, ptp_y, max(ptp_x,ptp_y)*0.05])
            elif ptp_x > 1e-9 and ptp_y > 1e-9: ax_3d.set_box_aspect([ptp_x, ptp_y, ptp_z])
        except Exception as e: print(f"Warning: 3D aspect ratio failed: {e}")
        plot_path_3d = os.path.join(image_dir, 'crossed_fields_3d.png')
        plt.savefig(plot_path_3d, dpi=300, bbox_inches='tight'); plt.close(fig_3d)
        print(f"Saved 3D plot to {plot_path_3d}")

        # === 2D Plot (XY Plane - Cycloid Focus) ===
        fig_2d, ax_2d = plt.subplots(figsize=(12, 4)) # Wider aspect ratio
        ax_2d.plot(positions[:, 0], positions[:, 1], color='red', linewidth=1.5, label='Particle Path (XY)')
        ax_2d.plot(positions[0, 0], positions[0, 1], 'go', markersize=7, label='Start')
        
        # Field Indicators (Simplified)
        plot_lim_x = ax_2d.get_xlim()
        plot_lim_y = ax_2d.get_ylim()
        indicator_scale = 0.05 * max(positions[:,0].max() - positions[:,0].min(), positions[:,1].max() - positions[:,1].min()) 
        # E field (Green arrow)
        ax_2d.arrow(positions[0,0], positions[0,1], 0, indicator_scale, 
                    color='green', width=indicator_scale*0.08, head_width=indicator_scale*0.3, 
                    length_includes_head=True, label='E field (+y)')
        # B field (Blue circle with dot)
        b_pos_x = positions[0,0] + indicator_scale * 0.5
        b_pos_y = positions[0,1] + indicator_scale * 0.5
        ax_2d.add_patch(plt.Circle((b_pos_x, b_pos_y), indicator_scale*0.15, color='blue', fill=False, lw=1.5))
        ax_2d.plot(b_pos_x, b_pos_y, 'bo', markersize=4) 
        ax_2d.text(b_pos_x + indicator_scale*0.2, b_pos_y, ' B (+z)', color='blue', va='center')

        ax_2d.set_xlabel('X Position (m) - Drift Direction')
        ax_2d.set_ylabel('Y Position (m)')
        ax_2d.set_title('Charged Particle E×B Drift (Cycloidal Motion - XY Plane)')
        ax_2d.legend(fontsize='small')
        ax_2d.grid(True, linestyle='--', alpha=0.5)
        ax_2d.axhline(0, color='black', linewidth=0.5); ax_2d.axvline(0, color='black', linewidth=0.5)
        ax_2d.ticklabel_format(style='sci', axis='both', scilimits=(-3,3))
        ax_2d.set_aspect('equal', adjustable='box') # Ensure true shape
        # Adjust limits slightly to ensure loops aren't cut off
        ax_2d.set_xlim(positions[:,0].min() - indicator_scale*0.5, positions[:,0].max() + indicator_scale*0.5)
        ax_2d.set_ylim(positions[:,1].min() - indicator_scale*0.5, positions[:,1].max() + indicator_scale*1.5) # Extra space for E arrow

        plot_path_2d = os.path.join(image_dir, 'crossed_fields.png')
        plt.savefig(plot_path_2d, dpi=300, bbox_inches='tight')
        print(f"Saved 2D XY-plot to {plot_path_2d}")
        plt.close(fig_2d)
        
    return positions, velocities, observations

def magnetic_bottle(q=ELECTRON_CHARGE, m=ELECTRON_MASS, B0=1.0, z_max=1.0, v0=1e6, v_parallel_ratio=0.3, steps=5000, dt_divisor=50.0, save_plots=True):
    """
    Simulate a charged particle in a magnetic bottle field.
    Field is approximated by B_z(z) = B0 * (1 + (z/z_max)^2), B_r approx -r/2 * dBz/dz.
    Dynamically calculates dt based on max expected cyclotron frequency.
    """
    # Estimate max field and min period to set dt
    B_max_approx = B0 * (1 + (z_max / z_max)**2) # Field at z=z_max
    if abs(q) > 1e-20 and B_max_approx > 1e-9:
        omega_c_max = abs(q * B_max_approx / m)
        T_cyc_min = 2 * np.pi / omega_c_max
        dt = T_cyc_min / dt_divisor
        print(f"Magnetic Bottle: Estimated B_max ~ {B_max_approx:.2f} T, T_cyc_min ~ {T_cyc_min:.3e} s. Using dt = {dt:.3e} s ({T_cyc_min:.1e} / {dt_divisor:.0f})")
    else:
        dt = 1e-12 # Fallback default dt if B or q are zero/tiny
        print(f"Warning: B0 or q is near zero, using fallback dt={dt:.3e} s for magnetic bottle.")

    # Define the magnetic field function B(r, z)
    def B_field(r_vec):
        x, y, z = r_vec
        rho = np.sqrt(x**2 + y**2)
        # Axial component
        Bz = B0 * (1 + (z / z_max)**2)
        # Radial component (approximation)
        dBz_dz = B0 * (2 * z / z_max**2)
        Br = - (rho / 2) * dBz_dz
        # Convert radial B back to x, y components
        Bx = Br * (x / rho) if rho > 1e-9 else 0
        By = Br * (y / rho) if rho > 1e-9 else 0
        return np.array([Bx, By, Bz])

    # Initial conditions
    pitch_angle = np.arccos(v_parallel_ratio)
    v_parallel = v0 * np.cos(pitch_angle)
    v_perp = v0 * np.sin(pitch_angle)
    
    # Start off-axis slightly to get radial field component effect
    r_initial = 0.1 * z_max # Start at 10% of z_max radially
    v0_vec = np.array([0.0, v_perp, v_parallel])  # Start with vy=v_perp
    r0_vec = np.array([r_initial, 0.0, 0.0])      # Start on x-axis at z=0

    E = np.array([0.0, 0.0, 0.0])  # No electric field
    
    # Run simulation (passing the B_field function)
    print(f"Running Magnetic Bottle simulation with {steps} steps...")
    positions, velocities = simulate_particle_motion(q, m, E, B_field, v0_vec, r0_vec, dt, steps)
    print("Simulation finished.")

    # --- Observations --- 
    # Magnetic moment (approximate conservation: mu = 0.5 * m * v_perp^2 / B)
    v_perp_mag = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
    B_mag_at_pos = np.array([np.linalg.norm(B_field(pos)) for pos in positions])
    magnetic_moment = 0.5 * m * v_perp_mag**2 / B_mag_at_pos
    avg_mu = np.mean(magnetic_moment[1:]) # Skip first point which might be noisy
    max_z = np.max(positions[:, 2])
    initial_pitch_angle_deg = np.degrees(pitch_angle)

    observations = {
        'magnetic_moment': f"Magnetic moment: {avg_mu:.3e} J/T (conserved)",
        'pitch_angle': f"Pitch angle: {initial_pitch_angle_deg:.1f}°",
        'max_z': f"Maximum z reached: {max_z:.3e} m"
    }
    for key, value in observations.items():
        print(value)
        
    # --- Plotting --- 
    if save_plots:
        print("Generating 3D trajectory plot...")
        # 3D Trajectory Plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        # Plot Particle Trajectory
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'red', linewidth=1.5, label='Particle Trajectory')
        ax.plot(positions[0, 0], positions[0, 1], positions[0, 2], 'go', markersize=6, label='Start')
        ax.plot(positions[-1, 0], positions[-1, 1], positions[-1, 2], 'ro', markersize=6, label='End')

        # Plot Central B-field Axis (Simplified representation)
        z_min_plot = np.min(positions[:, 2]) - 0.1 * np.ptp(positions[:, 2])
        z_max_plot = np.max(positions[:, 2]) + 0.1 * np.ptp(positions[:, 2])
        z_plot_axis = np.linspace(z_min_plot, z_max_plot, 2)
        ax.plot(np.zeros_like(z_plot_axis), np.zeros_like(z_plot_axis), z_plot_axis, 'b--', linewidth=1.5, label='B-field Axis')

        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title('Particle Trajectory in a Magnetic Bottle')
        ax.legend(fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.ticklabel_format(style='sci', axis='both', scilimits=(-2,2))
        
        # Adjust view and aspect ratio for clarity
        try:
            ax.set_box_aspect([np.ptp(positions[:,0]), np.ptp(positions[:,1]), np.ptp(positions[:,2])]) 
        except AttributeError:
             ax.set_aspect('auto') # Fallback for older matplotlib
        ax.view_init(elev=25, azim=-75) # Adjust view angle

        plot_path = os.path.join(image_dir, 'magnetic_bottle_trajectory.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved 3D plot to {plot_path}")
        plt.close(fig)
        
        print("Generating static field lines plot...")
        # Static Image of Magnetic Bottle Field (Simplified)
        fig_static, ax_static = plt.subplots(figsize=(8, 6))
        # Draw converging field lines schematically
        num_lines = 7
        z_static = np.linspace(-z_max * 1.1, z_max * 1.1, 100)
        y_max_static = z_max * 0.5 # Max radial extent for drawing lines
        for i in range(num_lines):
            # Vary radial position based on line index
            start_y = y_max_static * (i / (num_lines -1) - 0.5) * 2 if num_lines > 1 else 0
            # Field lines converge - approximate with a parabola shape inward
            x_line = z_static
            # Simple quadratic convergence towards z-axis
            y_line = start_y * (1 - 0.8*(z_static / z_max)**2)
            ax_static.plot(x_line, y_line, 'b-', alpha=0.6)
            # Add arrowheads (simple triangle markers)
            mid_idx = len(x_line) // 2
            end_idx = -1
            ax_static.plot(x_line[mid_idx], y_line[mid_idx], 'b>', markersize=6)
            ax_static.plot(x_line[end_idx], y_line[end_idx], 'b>', markersize=6) 

        ax_static.set_xlabel('Z Position (m)')
        ax_static.set_ylabel('Radial Position (Conceptual)')
        ax_static.set_title('Magnetic Bottle Field Lines (Schematic)')
        ax_static.grid(True, linestyle='--', alpha=0.3)
        ax_static.axhline(0, color='black', linewidth=0.5)
        ax_static.axvline(0, color='black', linewidth=0.5)
        ax_static.ticklabel_format(style='sci', axis='x', scilimits=(-2,2))
        ax_static.set_ylim(-y_max_static * 1.1, y_max_static * 1.1)

        plot_path_static = os.path.join(image_dir, 'magnetic_bottle_static.png')
        plt.savefig(plot_path_static, dpi=300, bbox_inches='tight')
        print(f"Saved static image to {plot_path_static}")
        plt.close(fig_static)
        
    return positions, velocities, observations

# Main execution
if __name__ == "__main__":
    print("Running Lorentz Force simulations...")
    
    # Uniform magnetic field simulation
    print("\n1. Simulating uniform magnetic field...")
    pos_uniform, vel_uniform, obs_uniform = uniform_magnetic_field(v_parallel=0.5e6)
    
    # Crossed fields simulation
    print("\n2. Simulating crossed E×B fields...")
    pos_crossed, vel_crossed, obs_crossed = crossed_fields()
    
    # Magnetic bottle simulation
    print("\n3. Simulating magnetic bottle...")
    pos_bottle, vel_bottle, obs_bottle = magnetic_bottle(steps=1500)
    
    print("\nAll simulations completed successfully!")
