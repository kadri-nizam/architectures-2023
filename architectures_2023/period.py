import logging
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.stats import ks_2samp

from architectures_2023.data import KeplerData
from architectures_2023.visual import PLOT_FORMAT, cdf, save_figure

__all__ = ["generate_figures"]


def generate_figures(kepler: KeplerData) -> None:
    ax = plt.subplot()

    logging.log(logging.INFO, "Generating figures for period related data.")

    cdf_singles_vs_multi_subsets(ax, kepler, "linear")
    cdf_singles_vs_multi_subsets(ax, kepler, "log")

    cdf_population(ax, kepler)

    cdf_with_ttv_flag(ax, kepler, "linear")
    cdf_with_ttv_flag(ax, kepler, "log")


def cdf_singles_vs_multi_subsets(
    ax: Axes, data: KeplerData, x_scale: str = "log"
) -> None:
    cdf(ax, data.singles["ttvperiod"], **PLOT_FORMAT["SINGLES"])
    cdf(ax, data.m2["ttvperiod"], **PLOT_FORMAT["M2"])
    cdf(ax, data.m3_plus["ttvperiod"], **PLOT_FORMAT["M3_PLUS"])

    ax.set_xscale(x_scale)  # type: ignore
    if x_scale == "linear":
        ax.set_xlim(0, 80)

    ax.set_xlabel("Period [days]")
    ax.set_ylabel("Normalized CDF")
    ax.legend()

    save_figure(ax, f"period_cdf_singles_vs_multi_subsets_{x_scale}.pdf")


def cdf_population(ax: Axes, data: KeplerData, x_scale: str = "log") -> None:
    # we further restrict the data to only candidates with a minimum of 3 transits
    logging.log(logging.INFO, f"Additional restriction: nttobs >= 3")
    data = KeplerData(
        data.singles.query("nttobs >= 3"),
        data.multis.query("nttobs >= 3"),
        status_flag=data.status_flag,
    )

    cdf(ax, data.all_candidates["ttvperiod"], **PLOT_FORMAT["ALL"])
    cdf(ax, data.singles["ttvperiod"], **PLOT_FORMAT["SINGLES"])
    cdf(ax, data.multis["ttvperiod"], **PLOT_FORMAT["MULTIS"])
    cdf(ax, data.m2["ttvperiod"], **PLOT_FORMAT["M2"])
    cdf(ax, data.m3_plus["ttvperiod"], **PLOT_FORMAT["M3_PLUS"])
    cdf(ax, data.innermost_multi["ttvperiod"], **PLOT_FORMAT["M_INNERMOST"])
    cdf(ax, data.middle_multi["ttvperiod"], **PLOT_FORMAT["M_MIDDLE"])
    cdf(ax, data.outermost_multi["ttvperiod"], **PLOT_FORMAT["M_OUTERMOST"])

    ax.set_xscale(x_scale)  # type: ignore
    if x_scale == "linear":
        ax.set_xlim(0, 80)

    ax.set_xlabel("Period [days]")
    ax.set_ylabel("Normalized CDF")

    # we'll have to break the legend into two sections because there are a lot of lines
    # See: https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html#multiple-legends-on-the-same-axes
    handles, labels = ax.get_legend_handles_labels()
    upper_left = ax.legend(handles[:5], labels[:5], loc="upper left")
    ax.legend(handles[5:], labels[5:], loc="lower right", markerfirst=False)
    ax.add_artist(upper_left)

    save_figure(ax, f"period_cdf_population_{x_scale}.pdf")


def cdf_with_ttv_flag(ax: Axes, data: KeplerData, x_scale: str = "log"):
    data = KeplerData(
        data.singles.query("nttobs >= 3"),
        data.multis.query("nttobs >= 3"),
        status_flag=data.status_flag,
    )
    logging.log(
        logging.INFO,
        f"Additional restriction for data without ttvflag filter: nttobs >= 3",
    )

    ttv_regex = r"t?1\d{2}|t?\d[12]\d|t?\d{2}[89]"
    data_w_ttv = KeplerData(
        data.singles.query("ttvflag.str.match(@ttv_regex)"),
        data.multis.query("ttvflag.str.match(@ttv_regex)"),
        status_flag=data.status_flag,
    )
    logging.log(
        logging.INFO,
        f"Additional restriction for data with ttvflag filter:"
        f"nttobs >= 3 and ttvflag must match regex {ttv_regex}",
    )

    # Some additional formatting for the plot is required here
    fmt = PLOT_FORMAT["SINGLES"].copy()
    fmt["label"] = f"{fmt['label']} w/o TTV"
    cdf(ax, data.singles["ttvperiod"], **fmt)

    fmt["label"] = fmt["label"].replace("w/o", "w/")
    fmt["linestyle"] = "dashed"
    cdf(ax, data_w_ttv.singles["ttvperiod"], **fmt)

    fmt = PLOT_FORMAT["MULTIS"].copy()
    fmt["label"] = f"{fmt['label']} w/o TTV"
    cdf(ax, data.multis["ttvperiod"], **fmt)

    fmt["label"] = fmt["label"].replace("w/o", "w/")
    fmt["linestyle"] = "dashed"
    cdf(ax, data_w_ttv.multis["ttvperiod"], **fmt)

    ax.set_xscale(x_scale)  # type: ignore
    if x_scale == "linear":
        ax.set_xlim(0, 80)

    ax.set_xlabel("Period [days]")
    ax.set_ylabel("Normalized CDF")
    ax.legend()

    save_figure(ax, f"period_cdf_with_ttv_flag_{x_scale}.pdf")
