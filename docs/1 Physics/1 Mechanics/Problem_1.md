# Problem 1: Investigating the Range as a Function of the Angle of Projection

## Motivation

Projectile motion, while seemingly simple, offers a rich playground for exploring fundamental principles of physics. The problem is straightforward: analyze how the range of a projectile depends on its angle of projection. Yet, beneath this simplicity lies a complex and versatile framework. The equations governing projectile motion involve both linear and quadratic relationships, making them accessible yet deeply insightful.

What makes this topic particularly compelling is the number of free parameters involved in these equations, such as initial velocity, gravitational acceleration, and launch height. These parameters give rise to a diverse set of solutions that can describe a wide array of real-world phenomena, from the arc of a soccer ball to the trajectory of a rocket.

## Theoretical Foundation

### Deriving the Equations of Motion

Let's start by deriving the equations of motion for a projectile from first principles. We'll consider a particle launched from the origin with an initial velocity $v_0$ at an angle $\theta$ with respect to the horizontal.

The initial velocity components are:
- Horizontal component: $v_{0x} = v_0 \cos\theta$
- Vertical component: $v_{0y} = v_0 \sin\theta$

Using Newton's second law and assuming no air resistance, the only force acting on the projectile is gravity. This gives us:

$\frac{d^2x}{dt^2} = 0$

$\frac{d^2y}{dt^2} = -g$

Integrating these equations with respect to time, we get:

$v_x = v_{0x} = v_0 \cos\theta$ (constant)

$v_y = v_{0y} - gt = v_0 \sin\theta - gt$

Integrating again to find the position:

$x(t) = v_0 \cos\theta \cdot t$

$y(t) = v_0 \sin\theta \cdot t - \frac{1}{2}gt^2$

### Family of Solutions

These equations represent a family of solutions parameterized by:
- Initial velocity magnitude $v_0$
- Launch angle $\theta$
- Gravitational acceleration $g$
- Initial height (if not launching from ground level)

By varying these parameters, we can generate a diverse set of trajectories, each representing a different physical scenario.

## Analysis of the Range

### Range as a Function of Angle

The range $R$ is the horizontal distance traveled when the projectile returns to its initial height. For a projectile launched from and landing on the same horizontal plane, we can find the time of flight $T$ by setting $y(T) = 0$:

$0 = v_0 \sin\theta \cdot T - \frac{1}{2}gT^2$

Solving for $T$ (excluding the trivial solution $T = 0$):

$T = \frac{2v_0 \sin\theta}{g}$

The range is then given by:

$R = x(T) = v_0 \cos\theta \cdot T = v_0 \cos\theta \cdot \frac{2v_0 \sin\theta}{g} = \frac{v_0^2 \sin(2\theta)}{g}$

This equation reveals that:
1. The range is proportional to the square of the initial velocity.
2. The range depends on the angle through the term $\sin(2\theta)$.
3. The maximum range occurs when $\sin(2\theta) = 1$, which happens when $\theta = 45°$.

### Influence of Other Parameters

- **Initial Velocity**: The range is proportional to $v_0^2$, so doubling the initial velocity quadruples the range.
- **Gravitational Acceleration**: The range is inversely proportional to $g$, so on planets with weaker gravity (e.g., the Moon), the same projectile would travel farther.
- **Launch Height**: If the projectile is launched from a height $h$ above the landing level, the range equation becomes more complex and the optimal angle shifts below 45°.

## Practical Applications

### Real-World Adaptations

1. **Uneven Terrain**: When launching from one elevation to another, the optimal angle deviates from 45°. For uphill trajectories, the optimal angle is greater than 45°, while for downhill trajectories, it's less than 45°.

2. **Air Resistance**: In reality, air resistance significantly affects projectile motion. It introduces a velocity-dependent force that typically reduces the range and lowers the optimal launch angle to about 40-43° for most sports projectiles.

3. **Spin Effects**: Many projectiles, like golf balls or footballs, experience lift forces due to spin (Magnus effect), which can dramatically alter their trajectories.

4. **Variable Gravity**: For very long-range projectiles, the variation of gravity with altitude becomes significant.

## Implementation

### Python Simulation

<details>
<summary>Click to expand/collapse the Python simulation code</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

# Create an animation of a projectile at the optimal angle
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, max_range * 1.1)
ax.set_ylim(0, max_range * 0.6)
ax.grid(True)
ax.set_xlabel('Horizontal Distance (m)')
ax.set_ylabel('Height (m)')
ax.set_title(f'Projectile Motion at Optimal Angle ({optimal_angle}°)')

x_opt, y_opt, _ = calculate_trajectory(v0, optimal_angle, g, h0)
line, = ax.plot([], [], 'b-', linewidth=2)
point, = ax.plot([], [], 'ro', markersize=8)

def init():
    line.set_data([], [])
    point.set_data([], [])
    return line, point

def animate(i):
    if i < len(x_opt):
        line.set_data(x_opt[:i+1], y_opt[:i+1])
        point.set_data(x_opt[i], y_opt[i])
    return line, point

anim = FuncAnimation(fig, animate, init_func=init, frames=len(x_opt), interval=20, blit=True)

plt.tight_layout()
plt.show()

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

plt.tight_layout()
plt.show()
```
</details>

<br>

### Visualization Results

The simulation produces several insightful visualizations that help us understand the relationship between launch angle and projectile range.

#### Range vs. Launch Angle

![Range vs Launch Angle](images/range_vs_angle.png)

This graph shows how the range varies with the launch angle. We can clearly see that the maximum range occurs at approximately 45° in the absence of air resistance, confirming our theoretical prediction. The curve follows the $\sin(2\theta)$ relationship derived earlier.

#### Trajectories for Different Launch Angles

![Trajectories for Different Launch Angles](images/trajectories.png)

This visualization displays the paths of projectiles launched at different angles (15°, 30°, 45°, 60°, and 75°). We can observe how lower angles result in flatter trajectories that cover more horizontal distance before reaching lower maximum heights, while higher angles produce more peaked paths with greater maximum heights but shorter ranges.

#### Effect of Initial Velocity on Range

![Range vs Launch Angle for Different Initial Velocities](images/range_vs_velocity.png)

This graph demonstrates how changes in initial velocity affect the range-angle relationship. As predicted by our equation $R = \frac{v_0^2 \sin(2\theta)}{g}$, the range increases with the square of the initial velocity. Note that while the magnitude of the range changes, the optimal angle remains at 45° for all initial velocities when launching from and landing on the same horizontal plane without air resistance.

#### Effect of Gravitational Acceleration on Range

![Range vs Launch Angle on Different Celestial Bodies](images/range_vs_gravity.png)

This visualization shows how the range-angle relationship changes under different gravitational conditions, simulating projectile motion on different celestial bodies. As expected from our equation, the range is inversely proportional to gravitational acceleration. On bodies with weaker gravity like the Moon and Mars, the same projectile travels much farther than on Earth, while on Jupiter with its stronger gravity, the range is significantly reduced.

## Limitations and Improvements

### Model Limitations

1. **Air Resistance**: The current model neglects air resistance, which significantly affects real-world projectiles. Including a drag force proportional to velocity or velocity squared would make the model more realistic.

2. **Wind Effects**: Wind can substantially alter projectile trajectories, especially for lightweight objects. A more comprehensive model would include wind as a horizontal force component.

3. **Lift Forces**: Many sports projectiles experience lift due to spin or asymmetric shape. These effects are not captured in the basic model.

4. **Earth's Curvature**: For very long-range projectiles, the curvature of the Earth becomes relevant.

### Suggested Improvements

1. **Drag Model**: Implement a drag force model using $F_d = -\frac{1}{2}\rho C_d A v^2$, where $\rho$ is air density, $C_d$ is the drag coefficient, $A$ is the cross-sectional area, and $v$ is velocity.

2. **Numerical Integration**: Use numerical methods like Runge-Kutta to solve the equations of motion when analytical solutions are not available (e.g., with air resistance).

3. **Monte Carlo Simulations**: Account for uncertainty in initial conditions by running Monte Carlo simulations to analyze the sensitivity of the range to small variations in parameters.

4. **3D Model**: Extend the model to three dimensions to account for lateral forces and movements.

## Conclusion

The study of projectile motion and the relationship between range and launch angle provides a fascinating window into the principles of classical mechanics. Through theoretical analysis and computational simulation, we've seen how the range depends on the launch angle, with a maximum at 45° under idealized conditions.

We've also explored how this relationship is influenced by other parameters such as initial velocity and gravitational acceleration, and how it changes under more realistic conditions. The provided Python implementation allows for further experimentation and visualization of these concepts.

This investigation not only deepens our understanding of a fundamental physics problem but also highlights the power of mathematical modeling in describing and predicting natural phenomena. The principles discussed here have wide-ranging applications, from sports and ballistics to space exploration and planetary science.