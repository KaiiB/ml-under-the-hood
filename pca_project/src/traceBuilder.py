
import json
import numpy as np
import pandas as pd
from src.dataBuilder import loadData, standardize
from src.pcaCore import computePCA

def build_trace_pca(path, dtype='numpy', seed=None, save_path="trace_pca.json"):
    """
    Builds a PCA trace JSON in the required format.

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
    eigvals, eigvecs, S = computePCA(data_std)
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
    # Steps (single step for PCA)
    # ----------------------------
    steps = [
        {
            "t": 0,
            "type": "pca",
            "payload": {
                "mean": mean_vector.tolist(),
                "covariance_matrix": S.tolist(),
                "eigenvalues": eigvals.tolist(),
                "eigenvectors": eigvecs.tolist(),
                "standardized_data": data_std.tolist(),   # you asked to include this
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
            "description": "From-scratch PCA demonstration",
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