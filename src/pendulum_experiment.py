import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

# Create the images directory if it doesn't exist
output_dir = os.path.join('docs', '1 Physics', '7 Measurements', 'images')
os.makedirs(output_dir, exist_ok=True)

def generate_pendulum_setup():
    """
    Generate a diagram of the pendulum experimental setup.
    """
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Draw support
    ax.plot([0, 2], [9, 9], 'k-', linewidth=3)
    
    # Draw string
    ax.plot([1, 1], [9, 4], 'k-', linewidth=1)
    
    # Draw bob
    circle = plt.Circle((1, 4), 0.3, fill=True, color='gray')
    ax.add_patch(circle)
    
    # Draw angle indicator
    ax.plot([1, 1.5], [9, 8], 'b--', alpha=0.5)
    ax.plot([1, 1], [9, 8], 'b--', alpha=0.5)
    ax.annotate('θ', xy=(1.15, 8.7), fontsize=12)
    
    # Draw length indicator
    ax.plot([1.2, 1.2], [9, 4], 'r-', linewidth=1)
    ax.plot([1.1, 1.3], [9, 9], 'r-', linewidth=1)
    ax.plot([1.1, 1.3], [4, 4], 'r-', linewidth=1)
    ax.annotate('L', xy=(1.3, 6.5), fontsize=12, color='red')
    
    # Add labels
    ax.annotate('Support', xy=(0.5, 9.2), fontsize=10)
    ax.annotate('String', xy=(0.7, 6.5), fontsize=10)
    ax.annotate('Bob', xy=(1, 3.7), fontsize=10)
    
    # Set axis properties
    ax.set_xlim(0, 2)
    ax.set_ylim(3, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Set title
    plt.title('Simple Pendulum Experimental Setup', fontsize=14)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'pendulum_setup.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_period_vs_length():
    """
    Generate a plot showing the relationship between pendulum period and length.
    """
    # Generate data points for different pendulum lengths
    lengths = np.linspace(0.3, 2.0, 10)  # Lengths from 0.3 to 2.0 meters
    g = 9.81  # Standard gravity in m/s^2
    
    # Calculate theoretical periods
    periods = 2 * np.pi * np.sqrt(lengths / g)
    
    # Add some random noise to simulate experimental data
    np.random.seed(42)  # For reproducibility
    measured_periods = periods + np.random.normal(0, 0.01, len(periods))
    
    # Calculate square root of length
    sqrt_lengths = np.sqrt(lengths)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(sqrt_lengths, measured_periods, color='blue', s=50, label='Measured data')
    
    # Calculate and plot best fit line
    slope, intercept, r_value, p_value, std_err = stats.linregress(sqrt_lengths, measured_periods)
    fit_line = slope * sqrt_lengths + intercept
    plt.plot(sqrt_lengths, fit_line, 'r-', label=f'Linear fit: T = {slope:.4f}√L + {intercept:.4f}')
    
    # Plot theoretical line
    theoretical_line = 2 * np.pi * sqrt_lengths / np.sqrt(g)
    plt.plot(sqrt_lengths, theoretical_line, 'g--', label=f'Theoretical: T = 2π√(L/g)')
    
    # Add error bars (assuming 0.5% uncertainty in period measurements)
    plt.errorbar(sqrt_lengths, measured_periods, yerr=measured_periods*0.005, fmt='none', capsize=3, color='blue', alpha=0.5)
    
    # Add labels and title
    plt.xlabel('Square Root of Length (√m)', fontsize=12)
    plt.ylabel('Period (s)', fontsize=12)
    plt.title('Relationship Between Pendulum Period and Length', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add text box with correlation coefficient
    textstr = f'Correlation coefficient (r²): {r_value**2:.6f}\nSlope: {slope:.6f} s/√m'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'pendulum_period_length.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_measurement_distribution():
    """
    Generate a histogram showing the distribution of period measurements.
    """
    # Sample data from the document
    measurements = np.array([20.12, 20.08, 20.15, 20.10, 20.13, 20.09, 20.11, 20.14, 20.07, 20.11])
    mean_value = np.mean(measurements)
    std_dev = np.std(measurements, ddof=1)  # Sample standard deviation
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create histogram
    n, bins, patches = plt.hist(measurements, bins=5, color='skyblue', alpha=0.7, edgecolor='black')
    
    # Add normal distribution curve
    x = np.linspace(mean_value - 4*std_dev, mean_value + 4*std_dev, 100)
    y = stats.norm.pdf(x, mean_value, std_dev) * len(measurements) * (bins[1] - bins[0])
    plt.plot(x, y, 'r-', linewidth=2, label='Normal distribution')
    
    # Add vertical line for mean
    plt.axvline(x=mean_value, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_value:.4f} s')
    
    # Add labels and title
    plt.xlabel('Time for 10 Oscillations (s)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Period Measurements', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add text box with statistics
    textstr = f'Mean: {mean_value:.4f} s\nStd Dev: {std_dev:.4f} s\nStd Error: {std_dev/np.sqrt(len(measurements)):.4f} s'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'pendulum_measurement_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Generating pendulum experiment visualizations...")
    
    # Generate all visualizations
    generate_pendulum_setup()
    print("Created pendulum setup diagram")
    
    generate_period_vs_length()
    print("Created period vs. length plot")
    
    generate_measurement_distribution()
    print("Created measurement distribution histogram")
    
    print(f"\nAll visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
