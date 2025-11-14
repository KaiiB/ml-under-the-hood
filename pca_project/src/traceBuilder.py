
import json
import numpy as np
import pandas as pd
from src.dataBuilder import loadData, standardize
from src.pcaCore import computePCA
from src.dataBuilder import generateGaussian

def buildTracePCA(path, dtype='numpy', seed=None, save_path="trace_pca.json"):
    """
    Builds a PCA trace JSON.

    Parameters
    ----------
    path : str
        Path to dataset (csv / pickle / npy)
    dtype : str
        'csv', 'pickle', or 'numpy'
    seed : int or None
        Optional seed for dataset-generation reproducibility upstream
        (PCA itself is deterministic)
    save_path : str
        Output JSON filename
    """

    # ----------------------------
    # Load and standardize data
    # ----------------------------
    data = loadData(path, dtype=dtype)
    data_std = standardize(data)

    mean_vector = data.mean(axis=0)
    n_samples, n_features = data.shape

    # ----------------------------
    # PCA computation
    # ----------------------------
    eigvals, eigvecs = computePCA(data_std)
    projected = data_std @ eigvecs

    # ----------------------------
    # Params blocks
    # ----------------------------
    params = {
        "dtype": dtype,
        "seed": seed,
        "path": path
    }

    params_full = {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "standardized": True,
        "input_dtype": dtype,
        "path": path
    }

    # ----------------------------
    # One-Shot PCA
    # ----------------------------
    steps = [
        {
            "t": 0,
            "type": "pca",
            "payload": {
                "mean": mean_vector.tolist(),
                "eigenvalues": eigvals.tolist(),
                "eigenvectors": eigvecs.tolist(),
                "projected_data": projected.tolist()
            }
        }
    ]

    # ----------------------------
    # Final trace object
    # ----------------------------
    trace = {
        "algo": "pca",
        "meta": {
            "description": "PCA demonstration",
            "version": "0.1"
        },
        "params": params,
        "params_full": params_full,
        "steps": steps
    }

    # ----------------------------
    # Save JSON
    # ----------------------------
    with open(save_path, "w") as f:
        json.dump(trace, f, indent=2)

    print(f"[✓] PCA trace saved to {save_path}")
    return trace

def buildTraceNumpyGeneraton(seed=42,
                             num_sets=1,
                             num_points=100,
                             dim=2,
                             mean=[0,0],
                             cov=[[1,0],[0,1]],
                             save_path="trace__gen_data.json"):
    """
    Builds a trace JSON for a random numpy array.

    Parameters
    ----------
    seed : int
        Seed for np.random functions
    num_sets : int
        number of random data sets/clusters to create
    num_points : int
        number of points to generate
    dim : int
        number of dimensions in dataset (default is 2)
    mean : array-like
        mean of the data to generate with
    cov : array-like
        covariance of the data to generate with
    save_path : str
        Output JSON filename
    """

    # ----------------------------
    # Generate Data
    # ----------------------------
    data = generateGaussian(num_sets=num_sets,
                            num_points=num_points,
                            dim=dim,
                            seed=seed,
                            mean=mean,
                            cov=cov)
        
    # ----------------------------
    # Params blocks
    # ----------------------------
    params = {
        "seed": seed,
        "num_sets": num_sets,
        "num_points": num_points,
        "dim": dim,
        "mean": mean,
        "cov": cov
    }

    params_full = {
        "sample_mean": data.mean(axis=0).tolist(),
        "sample_cov": np.cov(data).tolist(),
        "sample_std": data.std(axis=0).tolist(),
        "data": data.tolist()
    }

    # ----------------------------
    # Final trace object
    # ----------------------------
    trace = {
        "algo": "gaussianGenerate",
        "meta": {
            "description": "Generate Gaussian",
            "version": "0.1"
        },
        "params": params,
        "params_full": params_full
    }

    # ----------------------------
    # Save JSON
    # ----------------------------
    with open(save_path, "w") as f:
        json.dump(trace, f, indent=2)

    print(f"[✓] PCA trace saved to {save_path}")
    return trace