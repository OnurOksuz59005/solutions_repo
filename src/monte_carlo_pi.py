import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import time

# Create the images directory if it doesn't exist
output_dir = os.path.join('docs', '1 Physics', '6 Statistics', 'images')
os.makedirs(output_dir, exist_ok=True)

def estimate_pi_circle(num_points):
    """
    Estimate Pi using the circle-based Monte Carlo method.
    
    Args:
        num_points (int): Number of random points to generate
        
    Returns:
        tuple: (pi_estimate, points_inside, points_outside)
    """
    # Generate random points in a 2x2 square centered at the origin
    x = np.random.uniform(-1, 1, num_points)
    y = np.random.uniform(-1, 1, num_points)
    
    # Calculate distance from origin for each point
    distances = x**2 + y**2
    
    # Count points inside the unit circle (distance <= 1)
    inside_circle = distances <= 1
    points_inside = np.sum(inside_circle)
    points_outside = num_points - points_inside
    
    # Estimate Pi as 4 * (points inside circle / total points)
    pi_estimate = 4 * points_inside / num_points
    
    return pi_estimate, x[inside_circle], y[inside_circle], x[~inside_circle], y[~inside_circle]

def plot_circle_method(num_points, filename):
    """
    Plot the circle-based Monte Carlo method for estimating Pi.
    
    Args:
        num_points (int): Number of random points to generate
        filename (str): Filename to save the plot
    """
    pi_estimate, x_inside, y_inside, x_outside, y_outside = estimate_pi_circle(num_points)
    
    # Create plot
    plt.figure(figsize=(10, 10))
    
    # Plot the unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    plt.gca().add_patch(circle)
    
    # Plot the square
    plt.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], 'k-', linewidth=2)
    
    # Plot points inside the circle
    plt.scatter(x_inside, y_inside, color='blue', alpha=0.6, s=10, label='Inside')
    
    # Plot points outside the circle
    plt.scatter(x_outside, y_outside, color='red', alpha=0.6, s=10, label='Outside')
    
    # Set plot properties
    plt.axis('equal')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.grid(True, alpha=0.3)
    plt.title(f'Monte Carlo Estimation of pi using {num_points:,} points\nEstimate: {pi_estimate:.6f}', fontsize=14)
    plt.legend()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    return pi_estimate

def estimate_pi_buffon(num_needles, L=1, D=2):
    """
    Estimate Pi using Buffon's Needle method.
    
    Args:
        num_needles (int): Number of needles to drop
        L (float): Length of the needle
        D (float): Distance between parallel lines
        
    Returns:
        tuple: (pi_estimate, needle_positions, needle_angles, crossings)
    """
    # Generate random needle positions (center point y-coordinate)
    y_positions = np.random.uniform(0, D, num_needles)
    
    # Generate random needle angles (with horizontal)
    angles = np.random.uniform(0, np.pi, num_needles)
    
    # Calculate the y-extent of each needle (half-length * sin(angle))
    y_extents = 0.5 * L * np.sin(angles)
    
    # Determine if needles cross a line
    # A needle crosses a line if its y-position +/- y-extent crosses a multiple of D
    y_min = y_positions - y_extents
    y_max = y_positions + y_extents
    
    # Check if needle crosses the line at y=0 or y=D
    crossings = (y_min < 0) | (y_max > D) | (np.floor(y_min / D) != np.floor(y_max / D))
    num_crossings = np.sum(crossings)
    
    # Estimate Pi using Buffon's formula
    if num_crossings > 0:  # Avoid division by zero
        pi_estimate = (2 * L * num_needles) / (D * num_crossings)
    else:
        pi_estimate = np.nan
    
    return pi_estimate, y_positions, angles, crossings

def plot_buffon_method(num_needles, filename, L=1, D=2):
    """
    Plot Buffon's Needle method for estimating Pi.
    
    Args:
        num_needles (int): Number of needles to drop
        filename (str): Filename to save the plot
        L (float): Length of the needle
        D (float): Distance between parallel lines
    """
    pi_estimate, y_positions, angles, crossings = estimate_pi_buffon(num_needles, L, D)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot the parallel lines
    for i in range(5):  # Draw 5 lines
        plt.axhline(y=i*D, color='black', linewidth=2)
    
    # Plot a subset of needles (plotting all can be too cluttered)
    max_needles_to_plot = min(500, num_needles)  # Limit the number of needles to plot
    indices = np.random.choice(num_needles, max_needles_to_plot, replace=False)
    
    # Plot each needle
    for i in indices:
        y = y_positions[i]
        angle = angles[i]
        x_center = np.random.uniform(1, 9)  # Random x position for visualization
        
        # Calculate needle endpoints
        x1 = x_center - 0.5 * L * np.cos(angle)
        y1 = y - 0.5 * L * np.sin(angle)
        x2 = x_center + 0.5 * L * np.cos(angle)
        y2 = y + 0.5 * L * np.sin(angle)
        
        # Plot the needle (red if crossing a line, blue otherwise)
        if crossings[i]:
            plt.plot([x1, x2], [y1, y2], 'r-', linewidth=1.5, alpha=0.7)
        else:
            plt.plot([x1, x2], [y1, y2], 'b-', linewidth=1.5, alpha=0.7)
    
    # Set plot properties
    plt.xlim(0, 10)
    plt.ylim(-0.5, 4.5)
    plt.title(f"Buffon's Needle Method with {num_needles:,} needles\nEstimate: {pi_estimate:.6f}", fontsize=14)
    
    # Add legend
    plt.legend([Line2D([0], [0], color='red', linewidth=2), 
                Line2D([0], [0], color='blue', linewidth=2)], 
               ['Crossing a line', 'Not crossing'], loc='upper right')
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    return pi_estimate

def analyze_convergence(method, sample_sizes, num_trials=10):
    """
    Analyze the convergence of a Pi estimation method.
    
    Args:
        method (function): The estimation method to analyze
        sample_sizes (list): List of sample sizes to test
        num_trials (int): Number of trials for each sample size
        
    Returns:
        tuple: (sample_sizes, mean_estimates, std_estimates)
    """
    mean_estimates = []
    std_estimates = []
    
    for size in sample_sizes:
        estimates = []
        for _ in range(num_trials):
            if method.__name__ == 'estimate_pi_circle':
                pi_est, _, _, _, _ = method(size)
            else:  # Buffon's method
                pi_est, _, _, _ = method(size)
            estimates.append(pi_est)
        
        mean_estimates.append(np.mean(estimates))
        std_estimates.append(np.std(estimates))
    
    return sample_sizes, mean_estimates, std_estimates

def plot_convergence(method_name, sample_sizes, mean_estimates, std_estimates, filename):
    """
    Plot the convergence of a Pi estimation method.
    
    Args:
        method_name (str): Name of the method
        sample_sizes (list): List of sample sizes
        mean_estimates (list): Mean Pi estimates for each sample size
        std_estimates (list): Standard deviation of Pi estimates
        filename (str): Filename to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Plot the mean estimates
    plt.errorbar(sample_sizes, mean_estimates, yerr=std_estimates, fmt='o-', 
                 capsize=5, label='Estimated pi')
    
    # Plot the true value of Pi
    plt.axhline(y=np.pi, color='r', linestyle='--', label='True pi')
    
    # Set plot properties
    plt.xscale('log')
    plt.xlabel('Number of Samples', fontsize=12)
    plt.ylabel('Estimated pi', fontsize=12)
    plt.title(f'Convergence of {method_name} Method for Estimating pi', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add error bands
    plt.fill_between(sample_sizes, 
                     np.array(mean_estimates) - np.array(std_estimates),
                     np.array(mean_estimates) + np.array(std_estimates),
                     alpha=0.2)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def compare_methods(sample_sizes, circle_means, circle_stds, buffon_means, buffon_stds, filename):
    """
    Compare the convergence of both Pi estimation methods.
    
    Args:
        sample_sizes (list): List of sample sizes
        circle_means (list): Mean Pi estimates for circle method
        circle_stds (list): Standard deviation for circle method
        buffon_means (list): Mean Pi estimates for Buffon's method
        buffon_stds (list): Standard deviation for Buffon's method
        filename (str): Filename to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot the circle method
    plt.errorbar(sample_sizes, circle_means, yerr=circle_stds, fmt='o-', 
                 capsize=5, label='Circle Method')
    
    # Plot Buffon's method
    plt.errorbar(sample_sizes, buffon_means, yerr=buffon_stds, fmt='s-', 
                 capsize=5, label="Buffon's Needle Method")
    
    # Plot the true value of Pi
    plt.axhline(y=np.pi, color='r', linestyle='--', label='True pi')
    
    # Set plot properties
    plt.xscale('log')
    plt.xlabel('Number of Samples', fontsize=12)
    plt.ylabel('Estimated pi', fontsize=12)
    plt.title('Comparison of Monte Carlo Methods for Estimating pi', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add theoretical error bounds (proportional to 1/sqrt(N))
    x = np.array(sample_sizes)
    plt.plot(x, np.pi + 1/np.sqrt(x), 'k:', alpha=0.5)
    plt.plot(x, np.pi - 1/np.sqrt(x), 'k:', alpha=0.5)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Starting Monte Carlo pi estimation demonstration...")
    
    # Circle method visualizations
    print("Generating circle method visualizations...")
    circle_1k = plot_circle_method(1000, 'pi_circle_1000.png')
    circle_10k = plot_circle_method(10000, 'pi_circle_10000.png')
    circle_100k = plot_circle_method(100000, 'pi_circle_100000.png')
    
    print(f"Circle method estimates:")
    print(f"  1,000 points: pi ~ {circle_1k:.6f} (error: {abs(circle_1k - np.pi):.6f})")
    print(f"  10,000 points: pi ~ {circle_10k:.6f} (error: {abs(circle_10k - np.pi):.6f})")
    print(f"  100,000 points: pi ~ {circle_100k:.6f} (error: {abs(circle_100k - np.pi):.6f})")
    
    # Buffon's Needle method visualizations
    print("\nGenerating Buffon's Needle visualizations...")
    buffon_1k = plot_buffon_method(1000, 'pi_buffon_1000.png')
    buffon_10k = plot_buffon_method(10000, 'pi_buffon_10000.png')
    buffon_100k = plot_buffon_method(100000, 'pi_buffon_100000.png')
    
    print(f"Buffon's Needle method estimates:")
    print(f"  1,000 needles: pi ~ {buffon_1k:.6f} (error: {abs(buffon_1k - np.pi):.6f})")
    print(f"  10,000 needles: pi ~ {buffon_10k:.6f} (error: {abs(buffon_10k - np.pi):.6f})")
    print(f"  100,000 needles: pi ~ {buffon_100k:.6f} (error: {abs(buffon_100k - np.pi):.6f})")
    
    # Convergence analysis
    print("\nAnalyzing convergence rates...")
    sample_sizes = [100, 500, 1000, 5000, 10000, 50000]
    
    # Circle method convergence
    print("Analyzing circle method convergence...")
    circle_sizes, circle_means, circle_stds = analyze_convergence(estimate_pi_circle, sample_sizes)
    plot_convergence("Circle", circle_sizes, circle_means, circle_stds, 'pi_circle_convergence.png')
    
    # Buffon's Needle method convergence
    print("Analyzing Buffon's Needle method convergence...")
    buffon_sizes, buffon_means, buffon_stds = analyze_convergence(estimate_pi_buffon, sample_sizes)
    plot_convergence("Buffon's Needle", buffon_sizes, buffon_means, buffon_stds, 'pi_buffon_convergence.png')
    
    # Compare methods
    print("Comparing both methods...")
    compare_methods(sample_sizes, circle_means, circle_stds, buffon_means, buffon_stds, 'pi_methods_comparison.png')
    
    print("\nMonte Carlo pi estimation demonstration completed.")
    print(f"All plots saved to {output_dir}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
