import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from ipywidgets import interact
from src.dataBuilder import standardize, projection
from src.dataVisualization import createPlot, plotPoints, plotQuiver, plotEllipse, runPlotActions
from src.pcaCore import computePCA, pcaEllipse2D

def sliderFunction(angle, data, eigvals, eigvecs, mean, width, height, length_pc, _z=2.0, seed=42, drawEllipse=False, drawPrincipalComponents=True):
        W = eigvecs[:, :2] # Top 2 components

        # Create the plot
        fig, ax = createPlot(xlim=(-4, 4), ylim=(-4, 4), title='Can You Find the "Right" Principal Component?')

        # Define all actions
        actions = [
        {"func": plotPoints, "args": [data[:, 0], data[:, 1]], "kwargs": {"color": 'orange', "alpha": 0.6, "label": "Standardized Data"}},
        ]

        # Draw PCA ellipse if specified
        if drawEllipse:
            actions.append(
                  {"func": plotEllipse, "args": [mean, width, height, angle], "kwargs": {"edgecolor": 'black', "facecolor": 'none', "linewidth": 2}} # projection vector
                  )

        # Draw principal components if specified
        if drawPrincipalComponents:
            # Note that we calculated eigenvectors for (x-mu)(x-mu)^T, not (x-mu)(x-mu)^T/N, so we need to divide  by N to get the true eigenvalues of the covariance matrix
            # Moreover, we wuliply by 2 because sqrt(eigenvalue) gives variance along that component, so 2*sqrt(eigenvalue/N) includes 2 std devs to cover ~95% of data along that component
            actions.append(
                  {"func": plotQuiver, "args": [0,0, W[:, 0][0]*_z*np.sqrt(eigvals[0]/data.shape[0]), W[:, 0][1]*_z*np.sqrt(eigvals[0]/data.shape[0])],"kwargs": {"color": 'blue', "alpha": 0.6, "scale": 1, "label": "Eigenvector 1"}}) # eigevnvector 1
            actions.append(
                  {"func": plotQuiver, "args": [0,0, -W[:, 0][0]*_z*np.sqrt(eigvals[0]/data.shape[0]), -W[:, 0][1]*_z*np.sqrt(eigvals[0]/data.shape[0])], "kwargs": {"color": 'blue', "alpha": 0.6, "scale": 1}}) # eigevnvector 1
            actions.append(
                  {"func": plotQuiver, "args": [0,0, W[:, 1][0]*_z*np.sqrt(eigvals[1]/data.shape[0]), W[:, 1][1]*_z*np.sqrt(eigvals[1]/data.shape[0])], "kwargs": {"color": 'teal', "alpha": 0.6, "scale": 1, "label": "Eigenvector 2"}}) # eigevnvector 2
            actions.append(
                  {"func": plotQuiver, "args": [0,0, -W[:, 1][0]*_z*np.sqrt(eigvals[1]/data.shape[0]), -W[:, 1][1]*_z*np.sqrt(eigvals[1]/data.shape[0])], "kwargs": {"color": 'teal', "alpha": 0.6, "scale": 1}}) # eigevnvector 2

        # For our slider, we want to paraterize direction in degrees
        direction = np.deg2rad(angle)
        a = length_pc * np.cos(direction) 
        b = length_pc * np.sin(direction)

        # Because ax.quiver is a bit weird, we'll just draw two arrows in opposite directions for our principal components
        actions.append(
                {"func": plotQuiver, "args": [0, 0, a, b], 
                "kwargs": {"color": 'black', "scale": 1}})
        actions.append(
                {"func": plotQuiver, "args": [0, 0, -a, -b], 
                "kwargs": {"color": 'black', "scale": 1}})
        
        np.random.seed(seed) # for reproducibility
        for i in np.random.choice(np.arange(data.shape[0]), 20, replace=False):
                proj_i = projection(np.array([a, b]), data[i])
                actions.append(
                        {"func": plotQuiver, "args": [data[i][0], data[i][1], proj_i[0]-data[i][0], proj_i[1]-data[i][1],], "kwargs": {"scale": 1, "color": 'red'}} # residual vector
                )
                actions.append(
                        {"func": plotPoints, "args": [proj_i[0], proj_i[1]], "kwargs": {"color": 'green', "alpha": 0.6}} # projection points
                )
        # Run all actions
        runPlotActions(ax, actions, True)
        plt.show();