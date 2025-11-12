import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.patches import Ellipse
import numpy as np

def createPlot(xlim=None, ylim=None, title='Plot', show=True):
    
    if show == False:
        # Prevent from showing
        plt.ioff()

    # Create figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Set xlim and #ylim
    if xlim is not None:
        ax.set_xlim(*xlim);
    if ylim is not None:
        ax.set_ylim(*ylim);
    
    ax.set_aspect('equal');
    ax.spines['left'].set_position('zero');
    ax.spines['bottom'].set_position('zero');
    ax.spines['right'].set_color('none');
    ax.spines['top'].set_color('none');
    plt.title(title);
    plt.xlabel('X');
    plt.ylabel('Y');

    if show == False:
        # Prevent from showing
        plt.ion()
    return fig, ax

def plotPoints(ax, x, y, **kwargs):
    ax.scatter(x, y, **kwargs)
    
def plotQuiver(ax, x0, y0, u, v, **kwargs):
    ax.quiver(x0, y0, u, v, angles='xy', scale_units='xy', **kwargs)

def plotEllipse(ax, center, width, height, angle, **kwargs):
    ellipse = Ellipse(xy=center, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)

def runPlotActions(ax, actions, legend=True):
    for action in actions:
        func = action["func"]
        args = action.get("args", [])
        kwargs = action.get("kwargs", {})
        func(ax, *args, **kwargs)
    if legend:
        ax.legend()

def createPlotlyPlot(data=None, x=None, y=None, xlim=None, ylim=None, title='Plot'):
    if data is None or x is None or y is None:
        # Create an empty figure if no data is given
        fig = go.Figure()
    else:
        # Use px.scatter as the base figure
        fig = px.scatter(
            data_frame=data,
            x=x,
            y=y,
            title=title,
            labels={'x': 'X', 'y': 'Y'}
        )
    
    # Format layout
    fig.update_layout(
        title=title,
        xaxis=dict(
            range=xlim,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black'
        ),
        yaxis=dict(
            range=ylim,
            scaleanchor="x",  # equal aspect ratio
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black'
        ),
        template="simple_white",
        width=600,
        height=600,
        showlegend=True
    )

    return fig

def plotlyPoints(fig, x, y, name="Points", **kwargs):
    # default marker
    marker_defaults = dict(size=6)

    # extract custom marker if passed
    marker_custom = kwargs.pop("marker", {})
    
    # merge defaults and custom
    marker_combined = {**marker_defaults, **marker_custom}

    # if name is None, hide legend for this trace
    showlegend = True if name is not None else False

    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers',
        name=name,
        marker=marker_combined,
        showlegend=showlegend,
        **kwargs
    ))

def plotlyQuiver(fig, x0, y0, u, v, name=None, **kwargs):
    color = kwargs.get("color", "black")
    width = kwargs.get("arrowwidth", 1.5)
    size = kwargs.get("arrowsize", 1.5)
    head = kwargs.get("arrowhead", 3)
    # Draw arrows using annotations (these will stay visible)
    for xi, yi, ui, vi in zip(x0, y0, u, v):
        fig.add_annotation(
            x=xi + ui, y=yi + vi,
            ax=xi, ay=yi,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=head,
            arrowsize=size,
            arrowwidth=width,
            arrowcolor=color
        )
    # Add a invisible trace for legend only if a name is specified
    if name is not None:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color=color, width=width),
            name=name
        ))

def plotlyEllipse(fig, center, width, height, angle=0, name="Ellipse", **kwargs):
    cx, cy = center
    t = np.linspace(0, 2*np.pi, 200)
    x = (width / 2) * np.cos(t)
    y = (height / 2) * np.sin(t)
    
    # Apply rotation
    R = np.array([
        [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
        [np.sin(np.radians(angle)),  np.cos(np.radians(angle))]
    ])
    xy = np.dot(R, np.vstack([x, y]))
    x_rot, y_rot = xy[0] + cx, xy[1] + cy
    
    fig.add_trace(go.Scatter(
        x=x_rot, y=y_rot,
        mode='lines',
        name=name,
        line=dict(color=kwargs.get("color", "blue"))
    ))

def runPlotlyActions(fig, actions, legend=True):
    for action in actions:
        func = action["func"]
        args = action.get("args", [])
        kwargs = action.get("kwargs", {})
        func(fig, *args, **kwargs)
    if legend:
        fig.update_layout(showlegend=True)
    return fig