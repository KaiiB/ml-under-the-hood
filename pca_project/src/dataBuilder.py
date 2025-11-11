import numpy as np
import pandas as pd

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
    all_data = [] 
    for i in range(num_sets):
        data = np.random.multivariate_normal(mean, cov, size=num_points)
        all_data.append(data)

    return all_data

def getSample(data, n, seed=42):
    # Set seed for reproducibility
    np.random.seed(seed)
    subset_indices = np.random.choice(data.shape[0], n, replace=False)
    return data[subset_indices]

def standardize(data):
    # Standardize the data
    return (data - data.mean(axis=0)) / data.std(axis=0)

def projection(u, v):
    # Project data onto the given direction
    u_dot_v = np.dot(u, v)
    u_norm = np.linalg.norm(u)

    # Calculate projection
    projection = (u_dot_v / u_norm**2) * u
    
    return projection

def loadData(path, dtype=None):
    if dtype=='pandas':
        df = pd.read_csv(path, dtype=None)
        df = df.dropna()
        return df
    if dtype=='numpy':
        array = np.load(path)
        return array