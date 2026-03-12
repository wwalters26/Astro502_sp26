import pandas as pd
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler

def run_dpgmm_save_csv(csv_file, parameter_columns,
                       max_components=20, weight_threshold=1e-2,
                       random_state=0, output_csv="dpgmm_memberships.csv"):
    """
    Fit a DPGMM to the dataset and save a CSV with membership probabilities
    appended for each active cluster. Cluster weight is included in the column name.

    Parameters
    ----------
    csv_file : str
        Path to CSV containing the data.
    parameter_columns : list of str
        Column names to use for clustering.
    max_components : int
        Maximum number of components in the DPGMM.
    weight_threshold : float
        Minimum weight for a cluster to be considered active.
    random_state : int
        Random seed for reproducibility.
    output_csv : str
        Filename for the output CSV.

    Returns
    -------
    dict
        Dictionary containing the same outputs as the original run_dpgmm:
        - 'model', 'probs', 'weights', 'active_clusters'
    """
    # Load the dataset
    df = pd.read_csv(csv_file)
    X = df[parameter_columns].values

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit the DPGMM
    model = BayesianGaussianMixture(
        n_components=max_components,
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        max_iter=1000,
        random_state=random_state
    )
    model.fit(X_scaled)

    # Membership probabilities and weights
    probs = model.predict_proba(X_scaled)
    weights = model.weights_
    active_clusters = np.where(weights > weight_threshold)[0]

    # Prepare column names with cluster weights
    prob_columns = [f"Cluster_{i}_weight_{weights[i]:.4f}" for i in active_clusters]

    # Append membership probabilities to the original dataframe
    df_probs = pd.DataFrame(probs[:, active_clusters], columns=prob_columns)
    df_out = pd.concat([df, df_probs], axis=1)

    # Save to CSV
    df_out.to_csv(output_csv, index=False)

    print(f"Saved membership probabilities to {output_csv}")

    return {
        'model': model,
        'probs': probs,
        'weights': weights,
        'active_clusters': active_clusters
    }

fit = run_dpgmm_save_csv(
    csv_file="filename.csv",
    parameter_columns=["RA", "DEC", "PMRA", "PMDec", "Plx(mas)"],
    max_components=20,
    weight_threshold=0.01,
    random_state=42,
    output_csv="output.csv"
)