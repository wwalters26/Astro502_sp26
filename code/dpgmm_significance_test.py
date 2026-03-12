import pandas as pd
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# Function: run_dpgmm
# ------------------------------------------------------------
def run_dpgmm(X, max_components=20, weight_threshold=1e-2, random_state=0):
    """
    Fit a Dirichlet Process Gaussian Mixture Model (DPGMM) to a dataset
    and return relevant outputs including membership probabilities, cluster weights,
    and active clusters above a weight threshold.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (N, C) containing the dataset, where N = number of points
        and C = number of parameters (dimensions).
    max_components : int, default=20
        Maximum number of mixture components to allow in the DPGMM.
        The Dirichlet Process prior will suppress unused components.
    weight_threshold : float, default=0.01
        Minimum mixture weight for a cluster to be considered "active".
        Clusters with weights below this are treated as effectively non-existent.
    random_state : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
        - 'model' : fitted BayesianGaussianMixture object
        - 'probs' : array of shape (N, max_components), membership probabilities
        - 'weights' : array of mixture weights for all components
        - 'active_clusters' : indices of clusters with weight > weight_threshold
    """

    # Standardize each parameter to zero mean and unit variance
    # Important for Gaussian mixture models so no parameter dominates due to scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize and fit a Bayesian Gaussian Mixture model using the Dirichlet Process prior
    # This allows the model to effectively "choose" the number of clusters from the data
    model = BayesianGaussianMixture(
        n_components=max_components,                      # Upper limit on clusters
        covariance_type="full",                           # Full covariance matrices per component
        weight_concentration_prior_type="dirichlet_process",  # DP prior
        max_iter=1000,                                   # Maximum iterations for EM
        random_state=random_state
    )
    model.fit(X_scaled)

    # Compute membership probabilities for each point to each cluster
    # Each row sums to 1
    probs = model.predict_proba(X_scaled)

    # Extract the learned mixture weights for each cluster
    weights = model.weights_

    # Identify clusters that are "active", i.e., have weight above threshold
    active_clusters = np.where(weights > weight_threshold)[0]

    return {
        'model': model,
        'probs': probs,
        'weights': weights,
        'active_clusters': active_clusters
    }

# ------------------------------------------------------------
# Function: generate_null_data
# ------------------------------------------------------------
def generate_null_data(csv_file, parameter_columns):
    """
    Generate a Monte Carlo null dataset by sampling uniformly within
    an expanded range for each parameter: [min - std, max + std].
    This produces "faux" data with similar scale to the real dataset
    but without any real clustering.

    Parameters
    ----------
    csv_file : str
        Path to CSV containing the real data.
    parameter_columns : list of str
        Column names of the parameters (dimensions) to use.

    Returns
    -------
    null_data : np.ndarray
        Array of shape (N, C) of randomly generated null points.
    """
    # Load data from CSV
    df = pd.read_csv(csv_file)
    X = df[parameter_columns].values
    N, C = X.shape

    # Compute lower and upper bounds for each parameter
    # min - std and max + std gives a slightly wider range than the observed data
    mins = X.min(axis=0) - X.std(axis=0)
    maxs = X.max(axis=0) + X.std(axis=0)

    # Generate uniform random numbers within the expanded range for each parameter
    null_data = np.random.uniform(low=mins, high=maxs, size=(N, C))
    return null_data

# ------------------------------------------------------------
# Function: monte_carlo_cluster_test
# ------------------------------------------------------------
def monte_carlo_cluster_test(csv_file, parameter_columns, 
                              max_components=20, weight_threshold=1e-2, 
                              n_simulations=100, random_state=0):
    """
    Perform a Monte Carlo test to determine whether a cluster exists
    in a dataset using the DPGMM.

    Steps:
    1. Fit a DPGMM to the real data and record the largest cluster weight.
    2. Generate multiple null datasets (no clusters) and fit DPGMM to each.
    3. Compare the largest cluster weight in the real data to the distribution
       from null datasets to compute a p-value.

    Parameters
    ----------
    csv_file : str
        Path to CSV containing the real data.
    parameter_columns : list of str
        Column names for the C parameters to cluster on.
    max_components : int, default=20
        Maximum number of components in the DPGMM.
    weight_threshold : float, default=0.01
        Minimum weight for a cluster to be considered "active."
    n_simulations : int, default=100
        Number of null datasets to generate for the Monte Carlo test.
    random_state : int, default=0
        Seed for reproducibility.

    Returns
    -------
    p_value : float
        Monte Carlo p-value for the largest cluster.
    real_fit : dict
        Output of `run_dpgmm` applied to the real data.
    S_null : np.ndarray
        Array of largest cluster weights from each null simulation.
    """
    np.random.seed(random_state)

    # Load real data
    df = pd.read_csv(csv_file)
    X_real = df[parameter_columns].values
    N, C = X_real.shape

    # Fit DPGMM to the real data
    real_fit = run_dpgmm(X_real, max_components, weight_threshold, random_state)
    
    # If no clusters are active, return p-value = 1
    if len(real_fit['active_clusters']) == 0:
        print("No cluster found in real data.")
        return 1.0, real_fit, None

    # Use largest cluster weight as the cluster statistic
    S_real = max(real_fit['weights'][real_fit['active_clusters']])

    # Initialize null distribution of largest cluster weights
    S_null = []

    # Monte Carlo loop: generate null datasets and fit DPGMM to each
    for i in range(n_simulations):
        null_X = generate_null_data(csv_file, parameter_columns)
        null_fit = run_dpgmm(null_X, max_components, weight_threshold)
        
        if len(null_fit['active_clusters']) == 0:
            S_null.append(0.0)  # No clusters detected in null
        else:
            # Take the largest cluster weight from the null DPGMM
            S_null.append(max(null_fit['weights'][null_fit['active_clusters']]))

    S_null = np.array(S_null)

    # Compute Monte Carlo p-value: fraction of null datasets producing
    # a cluster weight >= the real largest cluster
    p_value = np.mean(S_null >= S_real)

    # Print summary for the user
    print(f"Real largest cluster weight: {S_real:.4f}")
    print(f"Monte Carlo p-value: {p_value:.4f}")
    print(f"Number of clusters found in real data: {len(real_fit['active_clusters'])}")

    return p_value, real_fit, S_null

# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
p_val, real_fit, null_weights = monte_carlo_cluster_test(
    csv_file="filename.csv",
    parameter_columns=["RA", "DEC", "PMRA", "PMDec", "Plx(mas)"],
    n_simulations=100
)