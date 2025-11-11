import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def createPlot(xlim=None, ylim=None, title='Plot'):
    # Create figure and axis
    fig, ax = plt.subplots()
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.set_aspect('equal')
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
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
        