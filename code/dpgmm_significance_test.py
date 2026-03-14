import pandas as pd
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
from joblib import Parallel, delayed


# ============================================================
# Function: run_dpgmm
# ============================================================
def run_dpgmm(X, max_components=20, weight_threshold=1e-2, random_state=0):
    """
    Fit a Dirichlet Process Gaussian Mixture Model (DPGMM) to the data.

    This function performs the core clustering step. It fits a Bayesian
    Gaussian mixture model with a Dirichlet Process prior, which allows
    the model to automatically infer the number of clusters present in
    the dataset.

    Parameters
    ----------
    X : ndarray (N x C)
        Input data matrix where:
        N = number of stars
        C = number of parameters (e.g., RA, Dec, PMRA, PMDec, Plx).

    max_components : int
        Maximum number of Gaussian components allowed in the mixture.
        The Dirichlet Process prior will typically deactivate many of
        these if they are unnecessary.

    weight_threshold : float
        Minimum mixture weight required for a component to be considered
        an "active cluster".

    random_state : int
        Seed used to make the clustering reproducible.

    Returns
    -------
    dict
        Dictionary containing:
        - model : fitted BayesianGaussianMixture object
        - probs : membership probabilities for each star (N x K)
        - weights : mixture weights for each Gaussian component
        - active_clusters : indices of clusters above weight_threshold
        - X_scaled : standardized data used during fitting
    """

    # --------------------------------------------------------
    # Standardize the parameters
    # --------------------------------------------------------
    # Gaussian mixture models are sensitive to parameter scale.
    # For example, RA might range over degrees while proper motions
    # may be much smaller. Standardizing ensures that no parameter
    # dominates purely due to units.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --------------------------------------------------------
    # Initialize Bayesian Gaussian Mixture model
    # --------------------------------------------------------
    # The Dirichlet Process prior allows the model to "turn off"
    # unnecessary clusters automatically.
    model = BayesianGaussianMixture(
        n_components=max_components,
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        max_iter=1000,
        random_state=random_state
    )

    # Fit model to standardized data
    model.fit(X_scaled)

    # --------------------------------------------------------
    # Extract results
    # --------------------------------------------------------

    # Membership probabilities for each star belonging to each cluster
    probs = model.predict_proba(X_scaled)

    # Mixture weights represent the fractional size of each Gaussian
    weights = model.weights_

    # Identify clusters that are meaningfully populated
    active_clusters = np.where(weights > weight_threshold)[0]

    return {
        'model': model,
        'probs': probs,
        'weights': weights,
        'active_clusters': active_clusters,
        'X_scaled': X_scaled
    }


# ============================================================
# Function: compute_primary_cluster_score
# ============================================================
def compute_primary_cluster_score(fit_result):
    """
    Compute a cluster strength score for the largest Gaussian component.

    The DPGMM often splits a real astrophysical cluster into several
    nearby Gaussians. This function compensates for that by allowing
    secondary Gaussians to contribute partially to the primary cluster.

    Conceptually:
    1. Identify the Gaussian with the largest mixture weight.
    2. Count all stars that belong to it probabilistically.
    3. For every other Gaussian:
       - Measure the distance between its center and the main cluster.
       - Estimate the probability of a point at that distance belonging
         to the main Gaussian.
       - Scale the membership of the secondary cluster by this probability.
    4. Add the contributions together.

    This boosts the score when the main cluster has been artificially
    split by the mixture model.

    Parameters
    ----------
    fit_result : dict
        Output dictionary from run_dpgmm()

    Returns
    -------
    float
        Normalized cluster score between 0 and 1 representing the
        effective membership fraction of the primary cluster.
    """

    model = fit_result['model']
    probs = fit_result['probs']
    X_scaled = fit_result['X_scaled']
    weights = fit_result['weights']

    means = model.means_
    covs = model.covariances_

    # --------------------------------------------------------
    # Identify the largest Gaussian component
    # --------------------------------------------------------
    idx_main = np.argmax(weights)

    mu_main = means[idx_main]
    cov_main = covs[idx_main]

    # Construct multivariate Gaussian distribution for the main cluster
    main_gaussian = multivariate_normal(mean=mu_main, cov=cov_main)

    # Number of stars
    N = probs.shape[0]

    # --------------------------------------------------------
    # Start with direct membership probabilities
    # --------------------------------------------------------
    # Sum probabilities that stars belong to the primary Gaussian
    total = np.sum(probs[:, idx_main])

    # --------------------------------------------------------
    # Add contributions from secondary Gaussians
    # --------------------------------------------------------
    for k in range(len(weights)):

        if k == idx_main:
            continue

        mu_k = means[k]

        # Euclidean distance between cluster centers
        dist = np.linalg.norm(mu_k - mu_main)

        # Create a hypothetical point located dist away
        test_point = mu_main.copy()
        test_point[0] += dist

        # Evaluate probability density of that point
        p_dist = main_gaussian.pdf(test_point)

        # Ignore extremely small contributions
        if p_dist < 1e-6:
            continue

        # Add scaled membership from the secondary cluster
        total += p_dist * np.sum(probs[:, k])

    # Normalize score by number of stars
    return total / N


# ============================================================
# Function: generate_null_data
# ============================================================
def generate_null_data(csv_file, parameter_columns):
    """
    Generate a synthetic null dataset for Monte Carlo testing.

    The goal is to create a dataset with similar parameter ranges
    but without real clustering structure.

    Method
    ------
    - Spatial parameters (RA, Dec, Plx) are randomized uniformly
      within an expanded bounding box of the real data.

    - Velocity parameters are also randomized within their observed
      ranges.

    This produces a "field-like" stellar population where stars are
    randomly distributed in the parameter space.

    Parameters
    ----------
    csv_file : str
        Path to input CSV file containing the real dataset.

    parameter_columns : list
        List of columns used for clustering.

    Returns
    -------
    ndarray
        Null dataset with same shape as the real data matrix.
    """

    df = pd.read_csv(csv_file)

    pos_cols = ["RA", "DEC", "Plx(mas)"]
    vel_cols = [c for c in parameter_columns if c not in pos_cols]

    # --------------------------------------------------------
    # Randomize spatial coordinates
    # --------------------------------------------------------
    pos_data = df[pos_cols].values
    pos_mins = pos_data.min(axis=0) - pos_data.std(axis=0)
    pos_maxs = pos_data.max(axis=0) + pos_data.std(axis=0)

    null_positions = np.random.uniform(
        low=pos_mins,
        high=pos_maxs,
        size=pos_data.shape
    )

    # --------------------------------------------------------
    # Randomize velocity parameters
    # --------------------------------------------------------
    vel_data = df[vel_cols].values
    vel_mins = vel_data.min(axis=0) - vel_data.std(axis=0)
    vel_maxs = vel_data.max(axis=0) + vel_data.std(axis=0)

    null_velocities = np.random.uniform(
        low=vel_mins,
        high=vel_maxs,
        size=vel_data.shape
    )

    # --------------------------------------------------------
    # Combine into final dataset
    # --------------------------------------------------------
    null_X = np.zeros((len(df), len(parameter_columns)))

    for i, c in enumerate(pos_cols):
        idx = parameter_columns.index(c)
        null_X[:, idx] = null_positions[:, i]

    for i, c in enumerate(vel_cols):
        idx = parameter_columns.index(c)
        null_X[:, idx] = null_velocities[:, i]

    return null_X


# ============================================================
# Function: monte_carlo_cluster_test
# ============================================================
def monte_carlo_cluster_test(csv_file, parameter_columns,
                             max_components=20, weight_threshold=1e-2,
                             n_simulations=100, random_state=0,
                             n_jobs=-1):
    """
    Perform a Monte Carlo significance test for clustering.

    The method compares the cluster score of the real dataset
    to a distribution of cluster scores produced from null datasets.

    Steps
    -----
    1. Fit the DPGMM to the real data.
    2. Compute the cluster score of the largest Gaussian.
    3. Generate many null datasets.
    4. Fit the same clustering model to each null dataset.
    5. Compute the cluster score for each null simulation.
    6. Estimate the p-value.

    p-value definition:
        Fraction of null simulations whose cluster score
        is greater than or equal to the real cluster score.

    Parameters
    ----------
    csv_file : str
        Path to dataset.

    parameter_columns : list
        Columns used for clustering.

    n_simulations : int
        Number of Monte Carlo trials.

    n_jobs : int
        Number of CPU cores used for parallel processing.

    Returns
    -------
    tuple
        (p_value, real_fit, S_null)
    """

    np.random.seed(random_state)

    df = pd.read_csv(csv_file)
    X_real = df[parameter_columns].values

    # Fit clustering model to real data
    real_fit = run_dpgmm(X_real, max_components, weight_threshold, random_state)

    if len(real_fit['active_clusters']) == 0:
        print("No cluster found in real data.")
        return 1.0, real_fit, None

    # Compute real cluster score
    S_real = compute_primary_cluster_score(real_fit)

    # --------------------------------------------------------
    # Single null simulation
    # --------------------------------------------------------
    def run_one_null(_):

        null_X = generate_null_data(csv_file, parameter_columns)

        null_fit = run_dpgmm(null_X, max_components, weight_threshold)

        if len(null_fit['active_clusters']) == 0:
            return 0.0

        return compute_primary_cluster_score(null_fit)

    # --------------------------------------------------------
    # Parallel Monte Carlo simulations
    # --------------------------------------------------------
    S_null = Parallel(n_jobs=n_jobs)(
        delayed(run_one_null)(i) for i in range(n_simulations)
    )

    S_null = np.array(S_null)

    # Compute p-value
    p_value = np.mean(S_null >= S_real)

    print(f"Real cluster score: {S_real:.4f}")
    print(f"Monte Carlo p-value: {p_value:.4f}")
    print(f"Number of clusters found in real data: {len(real_fit['active_clusters'])}")

    return p_value, real_fit, S_null


# ============================================================
# Example usage
# ============================================================

p_val, real_fit, null_weights = monte_carlo_cluster_test(
    csv_file="filename.csv",
    parameter_columns=["RA", "DEC", "PMRA", "PMDec", "Plx(mas)"],
    n_simulations=100
)