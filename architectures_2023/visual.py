from typing import Any

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

__all__ = ["cdf", "save_figure"]

# customize matplotlib rcParams upon import
mpl.rcParams["axes.labelsize"] = 10
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10
mpl.rcParams["legend.fontsize"] = 10
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Computer Modern Roman"]
mpl.rcParams["text.usetex"] = True


def cdf(
    ax: Axes,
    data: NDArray[np.floating[Any]],
    *,
    normalize_at_x: float | None = None,
    log_x: bool = False,
    log_y: bool = False,
    invert_x_axis: bool = False,
    **kwargs,
) -> Axes:
    """Plot the cumulative distribution function of the data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        The axes to plot the cdf on.
    data : numpy.ndarray
        The data to plot the cdf of.
    normalize_at_x : float, optional
        The x-value to normalize the cdf at. If not provided, the cdf will be normalized at the last x-value.
    log_x : bool, optional
        Whether to plot the x-axis on a log scale.
    log_y : bool, optional
        Whether to plot the y-axis on a log scale.
    invert_x_axis : bool, optional
        Whether to invert the x-axis.
    **kwargs
        Additional keyword arguments to pass to matplotlib.axes.Axes.plot.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the cdf was plotted on.
    """

    x = np.sort(data)
    y = np.arange(1, len(x) + 1)

    if not log_x and not log_y:
        x = np.insert(x, 0, 0)
        y = np.insert(y, 0, 0)

    # normalize the cdf w.r.t normalize_at_x if provided else
    # normalize at the last x value
    normalize_idx = np.argmax(x >= normalize_at_x) if normalize_at_x else -1
    y = y / y[normalize_idx]

    ax.plot(x, y, drawstyle="steps-post", **kwargs)

    if log_x:
        ax.set_xscale("log")

    if log_y:
        ax.set_yscale("log")

    if invert_x_axis:
        ax.invert_xaxis()

    return ax


def save_figure(f: Figure, filename: str, dpi: int = 300) -> None:
    """Save a matplotlib figure.

    Parameters
    ----------
    f : matplotlib.figure.Figure
        The figure to save.
    filename : str
        The filename to save the figure to.
    dpi : int, optional
        The resolution of the figure in dots per inch.

    Returns
    -------
    None
    """
    f.tight_layout(pad=0.1)
    f.savefig(filename, dpi=dpi)
