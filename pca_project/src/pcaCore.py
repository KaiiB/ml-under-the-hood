import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def computePCA(data):
    # Calculating covariance matrix and its eigenvalues/vectors
    S = data.T @ data # We don't need to normalize by n-1 for PCA; scalar multiples do not affect eigenvectors
    eigvals, eigvecs = np.linalg.eig(S) # eigenvalue decomposition

    # Sort by descending eigenvalue
    idx = np.argsort(eigvals)[::-1] # np.argsort sorts INDICES that WOULD SORT the array in ascending order; [::-1] reverses the order
    eigvals = eigvals[idx] # sorted eigenvalues
    eigvecs = eigvecs[:, idx] # sorted eigenvectors

    return eigvals, eigvecs

def pcaEllipse2D(eigvals, eigvecs, data, z=2.0):
    # Top 2 components
    W = eigvecs[:, :2] 

    # Compute ellipse parameters
    width = 2 * z * np.sqrt(eigvals[0]/data.shape[0])  # major axis
    height = 2 * z * np.sqrt(eigvals[1]/data.shape[0]) # minor axis

    # Compute rotation angle (in degrees)
    vx, vy = eigvecs[:, 0] # x and y components of the major axis
    angle = np.degrees(np.arctan2(vy, vx)) # angle of major axis
    mean = np.mean(data, axis=0) # data mean

    return mean, width, height