from enum import StrEnum
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy.typing import NDArray

__all__ = ["cdf", "save_figure", "Colors", "PLOT_FORMAT"]


def cdf(
    ax: Axes,
    data: Any,
    *,
    normalize_at_x: float | None = None,
    start_cdf_at: float = 0.0,
    include_zero: bool = True,
    **kwargs,
) -> Line2D:
    """Plot the cumulative distribution function of the data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        The axes to plot the cdf on.
    data : numpy.ndarray
        The data to plot the cdf of.
    normalize_at_x : float, optional
        The x-value to normalize the cdf at. If not provided, the cdf will be normalized at the last x-value.
    **kwargs
        Additional keyword arguments to pass to matplotlib.axes.Axes.plot.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the cdf was plotted on.
    """

    x = np.sort(data)
    y = np.arange(1, len(x) + 1)

    if include_zero:
        x = np.insert(x, 0, start_cdf_at)
        y = np.insert(y, 0, 0)

    # normalize the cdf w.r.t normalize_at_x if provided else
    # normalize at the last x value
    normalize_idx = np.argmax(x >= normalize_at_x) if normalize_at_x else -1
    y = y / y[normalize_idx]

    kwargs["label"] = kwargs["label"].format(len(data)) if "label" in kwargs else None
    (line_plot,) = ax.plot(x, y, drawstyle="steps-post", **kwargs)

    return line_plot


def save_figure(ax: Axes, filename: str, dpi: int = 300) -> None:
    """Save a matplotlib figure.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes we want to save the figure from.
    filename : str
        The filename to save the figure to.
    dpi : int, optional
        The resolution of the figure in dots per inch.

    Returns
    -------
    None
    """
    ax.figure.tight_layout(pad=0.1)
    ax.figure.savefig(filename, dpi=dpi)
    ax.clear()


# customize matplotlib rcParams upon import
mpl.rcParams.update(
    {
        "figure.figsize": (6, 4),
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.family": "cmu serif",
        "legend.frameon": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    }
)


# Custom color palette for the different lines we want to plot for this project
class Colors(StrEnum):
    ALL = "#444444"
    SINGLES = "#0C5DA5"
    MULTIS = "#FF2C00"
    M2 = "#845B97"
    M3_PLUS = "#00B945"
    M_INNERMOST = "#F46649"
    M_MIDDLE = "#BF2000"
    M_OUTERMOST = "#733022"


# Custom plot format for the different lines we want to plot for this project
PLOT_FORMAT = {
    "ALL": {
        "color": Colors.ALL,
        "linestyle": "solid",
        "linewidth": 1.5,
        "label": "[{}] All Planet Candidates",
    },
    "SINGLES": {
        "color": Colors.SINGLES,
        "linestyle": "solid",
        "linewidth": 1.5,
        "label": "[{}] Singles",
    },
    "MULTIS": {
        "color": Colors.MULTIS,
        "linestyle": "solid",
        "linewidth": 1.5,
        "label": "[{}] Multis",
    },
    "M2": {
        "color": Colors.M2,
        "linestyle": "dashed",
        "linewidth": 1,
        "label": r"[{}] $\mathcal{{M}}_2$",
    },
    "M3_PLUS": {
        "color": Colors.M3_PLUS,
        "linestyle": "dashed",
        "linewidth": 1,
        "label": r"[{}] $\mathcal{{M}}_{{3+}}$",
    },
    "M_INNERMOST": {
        "color": Colors.M_INNERMOST,
        "linestyle": "dotted",
        "linewidth": 1,
        "label": "[{}] Innermost of Multis",
    },
    "M_MIDDLE": {
        "color": Colors.M_MIDDLE,
        "linestyle": "dotted",
        "linewidth": 1,
        "label": "[{}] Middle of Multis",
    },
    "M_OUTERMOST": {
        "color": Colors.M_OUTERMOST,
        "linestyle": "dotted",
        "linewidth": 1,
        "label": "[{}] Outermost of Multis",
    },
}
