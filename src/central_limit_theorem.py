"""
Central Limit Theorem Simulation

This script demonstrates the Central Limit Theorem by simulating various probability distributions
and showing how their sampling distributions converge to normality as sample size increases.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# Create directory for images if it doesn't exist
image_dir = os.path.join('docs', '1 Physics', '6 Statistics', 'images')
os.makedirs(image_dir, exist_ok=True)

def generate_population(dist_type, size=100000, **params):
    """
    Generate a population from a specified distribution.
    
    Parameters:
    -----------
    dist_type : str
        Type of distribution ('uniform', 'exponential', 'binomial')
    size : int
        Size of the population to generate
    params : dict
        Additional parameters for the distribution
        
    Returns:
    --------
    numpy.ndarray
        Array of random values from the specified distribution
    """
    if dist_type == 'uniform':
        low = params.get('low', 0)
        high = params.get('high', 1)
        return np.random.uniform(low, high, size)
    
    elif dist_type == 'exponential':
        scale = params.get('scale', 1.0)  # 1/lambda
        return np.random.exponential(scale, size)
    
    elif dist_type == 'binomial':
        n = params.get('n', 10)
        p = params.get('p', 0.5)
        return np.random.binomial(n, p, size)
    
    else:
        raise ValueError(f"Distribution type '{dist_type}' not supported")

def sample_means(population, sample_size, num_samples):
    """
    Generate sample means from a population.
    
    Parameters:
    -----------
    population : numpy.ndarray
        The population to sample from
    sample_size : int
        Size of each sample
    num_samples : int
        Number of samples to take
        
    Returns:
    --------
    numpy.ndarray
        Array of sample means
    """
    means = np.zeros(num_samples)
    
    for i in range(num_samples):
        # Take a random sample from the population
        sample = np.random.choice(population, size=sample_size, replace=True)
        # Calculate and store the mean of this sample
        means[i] = np.mean(sample)
    
    return means

def plot_distribution(data, title, filename, bins=30, kde=True):
    """
    Plot a histogram of the data with a fitted normal distribution.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Data to plot
    title : str
        Plot title
    filename : str
        Filename to save the plot
    bins : int
        Number of bins for histogram
    kde : bool
        Whether to plot the kernel density estimate
    """
    plt.figure(figsize=(10, 6))
    
    # Plot histogram with KDE
    sns.histplot(data, bins=bins, kde=kde, stat='density')
    
    # Fit and plot normal distribution
    mu, sigma = np.mean(data), np.std(data)
    x = np.linspace(min(data), max(data), 1000)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, 
             label=f'Normal: μ={mu:.2f}, σ={sigma:.2f}')
    
    # Add vertical line for population mean
    plt.axvline(mu, color='k', linestyle='--', alpha=0.5, label=f'Mean: {mu:.2f}')
    
    # Add plot details
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(image_dir, filename))
    plt.close()

def plot_sampling_distributions(population, dist_name, sample_sizes=[5, 10, 30, 100], num_samples=10000):
    """
    Plot sampling distributions for different sample sizes.
    
    Parameters:
    -----------
    population : numpy.ndarray
        The population to sample from
    dist_name : str
        Name of the distribution (for plot titles and filenames)
    sample_sizes : list
        List of sample sizes to demonstrate
    num_samples : int
        Number of samples to take for each sample size
    """
    # Plot the population distribution
    plot_distribution(
        population, 
        f'Population Distribution: {dist_name}',
        f'{dist_name.lower()}_population.png'
    )
    
    # Calculate population parameters
    pop_mean = np.mean(population)
    pop_std = np.std(population)
    
    # Create a figure for comparing all sampling distributions
    plt.figure(figsize=(12, 10))
    
    # For each sample size, generate sampling distribution and plot
    for i, size in enumerate(sample_sizes):
        # Generate sample means
        means = sample_means(population, size, num_samples)
        
        # Plot individual sampling distribution
        plot_distribution(
            means,
            f'Sampling Distribution: {dist_name}, n={size}',
            f'{dist_name.lower()}_sampling_n{size}.png'
        )
        
        # Add to comparison plot
        plt.subplot(2, 2, i+1)
        sns.histplot(means, kde=True, stat='density')
        
        # Calculate theoretical standard error
        se = pop_std / np.sqrt(size)
        
        # Plot normal distribution with theoretical parameters
        x = np.linspace(min(means), max(means), 1000)
        plt.plot(x, stats.norm.pdf(x, pop_mean, se), 'r-', lw=2,
                label=f'Normal: μ={pop_mean:.2f}, σ={se:.4f}')
        
        plt.title(f'Sample Size: {size}')
        plt.xlabel('Sample Mean')
        plt.ylabel('Density')
        plt.legend()
    
    plt.suptitle(f'Sampling Distributions for {dist_name} Population', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the comparison figure
    plt.savefig(os.path.join(image_dir, f'{dist_name.lower()}_comparison.png'))
    plt.close()

def plot_convergence_qq(population, dist_name, sample_sizes=[5, 10, 30, 100], num_samples=10000):
    """
    Create QQ plots to show convergence to normality.
    
    Parameters:
    -----------
    population : numpy.ndarray
        The population to sample from
    dist_name : str
        Name of the distribution (for plot titles and filenames)
    sample_sizes : list
        List of sample sizes to demonstrate
    num_samples : int
        Number of samples to take for each sample size
    """
    plt.figure(figsize=(12, 10))
    
    for i, size in enumerate(sample_sizes):
        # Generate sample means
        means = sample_means(population, size, num_samples)
        
        # Create QQ plot
        plt.subplot(2, 2, i+1)
        stats.probplot(means, dist="norm", plot=plt)
        plt.title(f'QQ Plot: Sample Size {size}')
    
    plt.suptitle(f'Convergence to Normality: {dist_name} Population', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the QQ plot comparison
    plt.savefig(os.path.join(image_dir, f'{dist_name.lower()}_qq_plots.png'))
    plt.close()

def analyze_distribution(dist_type, dist_name, **params):
    """
    Analyze a distribution to demonstrate the Central Limit Theorem.
    
    Parameters:
    -----------
    dist_type : str
        Type of distribution ('uniform', 'exponential', 'binomial')
    dist_name : str
        Name of the distribution for plot titles
    params : dict
        Additional parameters for the distribution
    """
    print(f"Analyzing {dist_name} distribution...")
    
    # Generate population
    population = generate_population(dist_type, **params)
    
    # Plot sampling distributions
    plot_sampling_distributions(population, dist_name)
    
    # Plot QQ plots for convergence analysis
    plot_convergence_qq(population, dist_name)
    
    print(f"Completed analysis of {dist_name} distribution.")

def main():
    """Main function to run the CLT demonstration."""
    print("Starting Central Limit Theorem demonstration...")
    
    # Analyze uniform distribution
    analyze_distribution('uniform', 'Uniform', low=0, high=10, size=100000)
    
    # Analyze exponential distribution
    analyze_distribution('exponential', 'Exponential', scale=2.0, size=100000)
    
    # Analyze binomial distribution
    analyze_distribution('binomial', 'Binomial', n=10, p=0.3, size=100000)
    
    print("Central Limit Theorem demonstration completed.")
    print(f"All plots saved to {image_dir}")

if __name__ == "__main__":
    main()
