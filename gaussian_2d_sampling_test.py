import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.integrate as spi


def two_dimension_test():
    # Set parameters for the 2D normal distribution
    mu = np.array([0, 0])          # Mean of the normal distribution
    cov = np.array([[1, 0], [0, 1]])  # Covariance matrix (identity matrix)

    # Generate N values in logarithmic space
    N_values = np.logspace(1, 7, 10, dtype=int)

    # Initialize a list to store the corresponding errors
    errors = []

    # Define the 2D Gaussian PDF
    pdf = stats.multivariate_normal(mean=mu, cov=cov)

    # Loop over the logarithmic values of N
    for N in N_values:
        # Sample N values from the 2D normal distribution
        samples = np.random.multivariate_normal(mu, cov, N)
        
        # Generate a 2D histogram to count the samples in each bin
        bin_counts, x_edges, y_edges = np.histogram2d(samples[:,0], samples[:,1], bins=20, density=False)
        
        # Calculate the total number of samples (sanity check)
        total_samples = np.sum(bin_counts)
        
        # Initialize the total L1 norm error
        l1_error = 0
        
        # Loop over each bin to compute the analytic probability mass in that bin
        for i in range(len(x_edges) - 1):
            for j in range(len(y_edges) - 1):
                # Define the bin boundaries
                x_min, x_max = x_edges[i], x_edges[i+1]
                y_min, y_max = y_edges[j], y_edges[j+1]
                
                # Compute the analytic probability mass for the bin (integral of the PDF over the bin)
                # We approximate the integral by evaluating the PDF at the bin center and multiplying by the bin area
                bin_center_x = (x_min + x_max) / 2
                bin_center_y = (y_min + y_max) / 2
                bin_area = (x_max - x_min) * (y_max - y_min)
                # analytic_prob_mass = pdf.pdf([bin_center_x, bin_center_y]) * bin_area
                analytic_prob_mass = spi.dblquad(lambda x, y: pdf.pdf([x, y]), x_min, x_max, lambda x: y_min, lambda x: y_max)[0]
                # Get the sample count in this bin
                sample_count = bin_counts[i, j]
                
                # Compute the L1 norm error for this bin (absolute difference between sample count and analytic probability)
                l1_error += np.abs(sample_count/N - analytic_prob_mass)

        
        # Append the L1 norm error to the list
        errors.append(l1_error)

    # Convert to log10 values for linear plotting
    log_N_values = np.log10(N_values)
    log_errors = np.log10(errors)

    # Compute the theoretical line for N^-1/2
    theoretical_line = -0.5 * log_N_values + log_errors[0]

    # Plot the error as N increases in log10 scale
    plt.figure(figsize=(10, 6))
    plt.plot(log_N_values, log_errors, marker='o', linestyle='-', color='b', label='L1 Norm Error')
    plt.plot(log_N_values, theoretical_line, linestyle='--', color='r', label=r'$N^{-1/2}$')

    # Label the plot
    plt.xlabel(r'$\log_{10}(N)$')
    plt.ylabel(r'$\log_{10}(\mathrm{L1\ Norm\ Error})$')
    plt.title('Convergence of 2D Sample Distribution to Analytic PDF')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig("tests/plots/gaussian_2d_sampling_convergence.png")


def one_dimension_test():
    # Set parameters for the 1D normal distribution
    mu = 0          # Mean of the normal distribution
    sigma = 10       # Standard deviation of the normal distribution

    # Generate N values in logarithmic space
    N_values = np.logspace(1, 7, 10, dtype=int)

    # Initialize a list to store the corresponding errors
    errors = []

    # Loop over the logarithmic values of N
    for N in N_values:
        # Sample N values from the 1D normal distribution
        samples = np.random.normal(mu+4e-3, sigma, N)
        
        # Generate a histogram to count the samples in each bin
        bin_counts, bin_edges = np.histogram(samples, bins=20, density=False)
        
        # Calculate the total number of samples (sanity check)
        total_samples = np.sum(bin_counts)
        
        # Initialize the total L1 norm error
        l1_error = 0

        # set the error to the KS Statistic between the sample and the normal distribution
        l1_error = stats.kstest(samples, 'norm', args=(mu, sigma)).statistic
        
        # # Loop over each bin to compute the analytic probability mass in that bin
        # for i in range(len(bin_edges) - 1):
        #     # Define the bin boundaries
        #     x_min, x_max = bin_edges[i], bin_edges[i+1]
            
        #     # Compute the analytic probability mass for the bin (integral of the PDF over the bin)
        #     # We approximate the integral by evaluating the PDF at the bin center and multiplying by the bin width
        #     bin_center = (x_min + x_max) / 2
        #     bin_width = x_max - x_min
        #     analytic_prob_mass = spi.quad(lambda x: (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2), x_min, x_max)[0]
            
        #     # Get the sample count in this bin
        #     sample_count = bin_counts[i]
            
        #     # Compute the L1 norm error for this bin (absolute difference between sample count and analytic probability)
        #     l1_error += np.abs(sample_count/N - analytic_prob_mass)

        # Append the L1 norm error to the list
        errors.append(l1_error)

    # Convert to log10 values for linear plotting
    log_N_values = np.log10(N_values)
    log_errors = np.log10(errors)

    # Compute the theoretical line for N^-1/2
    theoretical_line = -0.5 * log_N_values + log_errors[0]

    # Plot the error as N increases in log10 scale
    plt.figure(figsize=(10, 6))
    plt.plot(log_N_values, log_errors, marker='o', linestyle='-', color='b', label='L1 Norm Error')
    plt.plot(log_N_values, theoretical_line, linestyle='--', color='r', label=r'$N^{-1/2}$')

    # Label the plot
    plt.xlabel(r'$\log_{10}(N)$')
    plt.ylabel(r'$\log_{10}(\mathrm{L1\ Norm\ Error})$')
    plt.title('Convergence of 1D Sample Distribution to Analytic PDF')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig("tests/plots/gaussian_1d_sampling_convergence.png")

if __name__ == '__main__':
    two_dimension_test()    
    # one_dimension_test()