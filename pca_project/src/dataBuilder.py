import numpy as np
import pandas as pd
from scipy.stats import zscore

def generateGaussian(num_sets=1, num_points=100, dim=2, seed=42, mean=None, cov=None):
    # Set covariance matrix to identity matrix if not provided

    if cov is None:
        cov = np.identity(dim)

    # Set mean vector to zero vector if not provided
    if mean is None:
        mean = np.zeros(dim)

    # Validate dimensions
    if len(cov) != dim:
        raise ValueError(f"Covariance matrix shape {len(cov)} does not match specified dimension {dim}")
    if len(mean) != dim:
        raise ValueError(f"Mean vector length {len(mean)} does not match specified dimension {dim}")
    
    # Set seed for reproducibility
    np.random.seed(seed)

    # Generate the datasets
    all_data = np.array([])
    for i in range(num_sets):
        data = np.random.multivariate_normal(mean, cov, size=num_points)
        all_data = np.append(all_data, data)

    return all_data

def getSample(data, n, seed=42):
    # Set seed for reproducibility
    np.random.seed(seed)
    subset_indices = np.random.choice(data.shape[0], n, replace=False)
    return data[subset_indices]

def standardize(data):
    # Standardize the data
    return (data - data.mean(axis=0)) / data.std(ddof=0, axis=0) # Little caveat: pandas calculates sample std; to make it consistent with numpy, set ddof=0 for pop std 

def projection(u, v):
    # Project data onto the given direction
    u_dot_v = np.dot(u, v)
    u_norm = np.linalg.norm(u)

    # Calculate projection
    projection = (u_dot_v / u_norm**2) * u
    
    return projection

def zPrune(df):
    # Copy + mean-center
    df_copy = df.copy() # we copy to avoid modifying the original dataframe

    # Compute z-scores for each numerical column (outlier deletion step 1)
    z_scores = np.abs(zscore(df_copy, nan_policy='omit'))

    # Set threshold for outlier detection (outlier deletion part 2)
    threshold = 3

    # Keep only rows where all z-scores are below the threshold (outlier deletion step 3)
    df_copy = df_copy[(z_scores < threshold).all(axis=1)]
    return df_copy

def loadData(path, dtype=None):
    if dtype=='csv':
        df = pd.read_csv(path, dtype=None)
        df = df.dropna()
        return df
    elif dtype=='pickle':
        df = pd.read_pickle(path)
        return df
    elif dtype=='numpy':
        array = np.load(path)
        return array