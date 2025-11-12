import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import plotly.graph_objects as go
from ipywidgets import interact
from src.dataBuilder import projection
from src.dataVisualization import plotPoints, plotQuiver, plotEllipse, runPlotActions
from src.dataVisualization import plotlyPoints, plotlyQuiver, plotlyEllipse, runPlotlyActions

def makeSliderFunction(fig, ax):

    initial_xlim = ax.get_xlim()
    initial_ylim = ax.get_ylim()
    initial_title = ax.get_title()
    initial_aspect = ax.get_aspect()

    def sliderFunction(
            angle,
            data,
            eigvals,
            eigvecs,
            mean,
            width,
            height,
            length_pc,
            _z=2.0,
            seed=42,
            drawEllipse=False,
            drawPrincipalComponents=True
            ):
            
            # Top 2 components
            W = eigvecs[:, :2]

            # Clear old plot, reuse same figure
            ax.clear()

            # Reapply the axis settings from the original createPlot call
            ax.set_xlim(initial_xlim)
            ax.set_ylim(initial_ylim)
            ax.set_aspect(initial_aspect if initial_aspect is not None else 'equal')

            # reapply spines/labels/title exactly as createPlot does
            ax.spines['left'].set_position('zero')
            ax.spines['bottom'].set_position('zero')
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.set_title(initial_title)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

            # Define all actions
            actions = [
            {"func": plotPoints, "args": [data[:, 0], data[:, 1]], "kwargs": {"color": 'orange', "alpha": 0.6, "label": "Standardized Data"}},
            ]

            # Draw PCA ellipse if specified
            if drawEllipse:
                actions.append(
                    {"func": plotEllipse, "args": [mean, width, height, angle], "kwargs": {"edgecolor": 'black', "facecolor": 'none', "linewidth": 2, "label": "Eigenvector 1"}} # projection vector
                    )

            # Draw principal components if specified
            if drawPrincipalComponents:
                # Note that we calculated eigenvectors for (x-mu)(x-mu)^T, not (x-mu)(x-mu)^T/N, so we need to divide by
                # # N to get the true eigenvalues of the covariance matrix.
                # Moreover, we wuliply by 2 because sqrt(eigenvalue) gives variance along that component,
                # so 2*sqrt(eigenvalue/N) includes 2 std devs to cover ~95% of data along that component.

                actions.append(
                    {"func": plotQuiver, "args": [0,0,
                                                    W[:, 0][0]*_z*np.sqrt(eigvals[0]/data.shape[0]),
                                                    W[:, 0][1]*_z*np.sqrt(eigvals[0]/data.shape[0])],
                                                    "kwargs": {"color": 'blue', "alpha": 0.6, "scale": 1, "label": "Eigenvector 1"}}
                                                    ) # eigevnvector 1
                actions.append(
                    {"func": plotQuiver, "args": [0,0,
                                                    -W[:, 0][0]*_z*np.sqrt(eigvals[0]/data.shape[0]),
                                                    -W[:, 0][1]*_z*np.sqrt(eigvals[0]/data.shape[0])],
                                                    "kwargs": {"color": 'blue', "alpha": 0.6, "scale": 1}}
                                                    ) # eigevnvector 1
                actions.append(
                    {"func": plotQuiver, "args": [0,0,
                                                    W[:, 1][0]*_z*np.sqrt(eigvals[1]/data.shape[0]),
                                                    W[:, 1][1]*_z*np.sqrt(eigvals[1]/data.shape[0])],
                                                    "kwargs": {"color": 'teal', "alpha": 0.6, "scale": 1, "label": "Eigenvector 2"}}
                                                    ) # eigevnvector 2
                actions.append(
                    {"func": plotQuiver, "args": [0,0,
                                                    -W[:, 1][0]*_z*np.sqrt(eigvals[1]/data.shape[0]),
                                                    -W[:, 1][1]*_z*np.sqrt(eigvals[1]/data.shape[0])],
                                                    "kwargs": {"color": 'teal', "alpha": 0.6, "scale": 1}}
                                                    ) # eigevnvector 2

            # For our slider, we want to paraterize direction in degrees.
            direction = np.deg2rad(angle)
            a = length_pc * np.cos(direction) 
            b = length_pc * np.sin(direction)

            # Because ax.quiver is a bit weird, we'll just draw two arrows in opposite directions for our principal components.
            actions.append(
                    {"func": plotQuiver, "args": [0, 0, a, b], 
                    "kwargs": {"color": 'black', "scale": 1, "label": "User Direction"}}
                    )
            actions.append(
                    {"func": plotQuiver, "args": [0, 0, -a, -b], 
                    "kwargs": {"color": 'black', "scale": 1}}
                    )
            
            # Draw residual vectors and projection points
            np.random.seed(seed) # for reproducibility
            for i in np.random.choice(np.arange(data.shape[0]), 20, replace=False):
                    proj_i = projection(np.array([a, b]), data[i])
                    # Residual vectors
                    actions.append(
                            {"func": plotQuiver, "args": [data[i][0], data[i][1], proj_i[0]-data[i][0], proj_i[1]-data[i][1],], "kwargs": {"scale": 1, "color": 'red'}}
                    )
                    # Projection points
                    actions.append(
                            {"func": plotPoints, "args": [proj_i[0], proj_i[1]], "kwargs": {"color": 'green', "alpha": 0.6}}
                    )
            # Run all actions
            runPlotActions(ax, actions, True)

            # Efficient redraw
            fig.canvas.draw_idle()

            return fig
    return sliderFunction

def makePlotlySliderFunction(fig):
    # Store the initial layout settings to reapply each update
    initial_layout = fig.layout.to_plotly_json()

    def sliderFunctionPlotly(
            angle,
            data,
            eigvals,
            eigvecs,
            mean,
            width,
            height,
            length_pc,
            _z=2.0,
            seed=42,
            drawEllipse=False,
            drawPrincipalComponents=True
        ):
        # Copy top 2 eigenvectors
        W = eigvecs[:, :2]

        # Clear previous traces and annotations
        fig.data = []
        fig.layout.annotations = []

        # Reapply layout settings (x/y limits, aspect ratio, etc.)
        fig.update_layout(
            xaxis=initial_layout["xaxis"],
            yaxis=initial_layout["yaxis"],
            title=initial_layout["title"],
            template=initial_layout["template"],
            width=initial_layout["width"],
            height=initial_layout["height"],
            showlegend=True
        )

        # Define all plot actions
        actions = [
            {"func": plotlyPoints, "args": [data[:, 0], data[:, 1]], "kwargs": {"name": "Standardized Data", "marker": {"color": "orange", "opacity": 0.6}}},
        ]

        # Draw PCA ellipse if specified
        if drawEllipse:
            actions.append(
                {"func": plotlyEllipse, "args": [mean, width, height, angle], "kwargs": {"color": "black", "name": "Covariance Ellipse"}}
            )

        # Draw principal components if specified
        if drawPrincipalComponents:
            # Scaling factor
            scale_1 = _z * np.sqrt(eigvals[0] / data.shape[0])
            scale_2 = _z * np.sqrt(eigvals[1] / data.shape[0])

            # Eigenvector 1 (both directions)
            actions.append({"func": plotlyQuiver, "args": [[0], [0], [W[0, 0] * scale_1], [W[1, 0] * scale_1]], "kwargs": {"color": "blue", "name": "Eigenvector 1"}})
            actions.append({"func": plotlyQuiver, "args": [[0], [0], [-W[0, 0] * scale_1], [-W[1, 0] * scale_1]], "kwargs": {"color": "blue"}})

            # Eigenvector 2 (both directions)
            actions.append({"func": plotlyQuiver, "args": [[0], [0], [W[0, 1] * scale_2], [W[1, 1] * scale_2]], "kwargs": {"color": "teal", "name": "Eigenvector 2"}})
            actions.append({"func": plotlyQuiver, "args": [[0], [0], [-W[0, 1] * scale_2], [-W[1, 1] * scale_2]], "kwargs": {"color": "teal"}})

        # User-selected direction
        direction = np.deg2rad(angle)
        a, b = length_pc * np.cos(direction), length_pc * np.sin(direction)
        actions.append({"func": plotlyQuiver, "args": [[0], [0], [a], [b]], "kwargs": {"color": "black", "name": "User Direction"}})
        actions.append({"func": plotlyQuiver, "args": [[0], [0], [-a], [-b]], "kwargs": {"color": "black"}})

        # Draw projection points and residuals
        np.random.seed(seed)
        sample_idx = np.random.choice(np.arange(data.shape[0]), 20, replace=False)
        for i in sample_idx:
            proj_i = projection(np.array([a, b]), data[i])
            # Residual vectors
            actions.append({"func": plotlyQuiver, "args": [[data[i][0]], [data[i][1]], [proj_i[0] - data[i][0]], [proj_i[1] - data[i][1]]], "kwargs": {"color": "red"}})
            # Projection points
            actions.append({"func": plotlyPoints, "args": [[proj_i[0]], [proj_i[1]]], "kwargs": {"name": None, "marker": {"color": "green", "opacity": 0.6}}})

        # Execute all plotting actions
        runPlotlyActions(fig, actions, legend=True)

        return fig

    return sliderFunctionPlotly