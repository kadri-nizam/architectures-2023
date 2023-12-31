import json
import logging

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy import stats

from architectures_2023.data import KeplerData
from architectures_2023.visual import PLOT_FORMAT, cdf, save_figure

__all__ = ["generate_figures"]


def generate_figures(kepler: KeplerData) -> None:
    logging.log(
        logging.INFO, f"Generating statistics for period related data with {kepler}"
    )
    stat_info = {
        "KS-Singles_vs_Multis": stats.ks_2samp(
            kepler.singles["ttvperiod"], kepler.multis["ttvperiod"]
        ),
        "KS-M2_vs_M3+": stats.ks_2samp(
            kepler.m2["ttvperiod"], kepler.m3_plus["ttvperiod"]
        ),
    }
    logging.log(
        logging.INFO,
        json.dumps(stat_info, indent=4, default=str),
    )

    logging.log(
        logging.INFO, f"Generating figures for period related data with\n{kepler}"
    )
    ax = plt.subplot()

    cdf_population(ax, kepler)
    cdf_singles_vs_multi_subsets(ax, kepler, x_scale="linear")
    cdf_singles_vs_multi_subsets(ax, kepler, x_scale="log")

    cdf_with_ttv_flag(ax, kepler, x_scale="linear")
    cdf_with_ttv_flag(ax, kepler, x_scale="log")

    fraction_of_transiting_companions(ax, kepler, large_planet_cutoff=5.0)


def cdf_singles_vs_multi_subsets(
    ax: Axes, data: KeplerData, *, x_scale: str = "log"
) -> None:
    cdf(
        ax,
        data.singles["ttvperiod"],
        include_zero=(x_scale == "linear"),
        **PLOT_FORMAT["SINGLES"],
    )
    cdf(
        ax,
        data.m2["ttvperiod"],
        include_zero=(x_scale == "linear"),
        **PLOT_FORMAT["M2"],
    )
    cdf(
        ax,
        data.m3_plus["ttvperiod"],
        include_zero=(x_scale == "linear"),
        **PLOT_FORMAT["M3_PLUS"],
    )

    ax.set_xscale(x_scale)  # type: ignore
    if x_scale == "linear":
        ax.set_xlim(0, 80)

    ax.set_xlabel("Period [days]")
    ax.set_ylabel("Normalized CDF")
    if x_scale == "log":
        ax.legend(loc="lower right")

    save_figure(ax, f"period_cdf_singles_vs_multi_subsets_{x_scale}.pdf")


def cdf_population(ax: Axes, data: KeplerData, *, x_scale: str = "log") -> None:
    # we further restrict the data to only candidates with a minimum of 3 transits
    logging.log(logging.INFO, f"Additional restriction: nttobs >= 3")
    data = KeplerData(
        data.singles.query("nttobs >= 3"),
        data.multis.query("nttobs >= 3"),
        status_flag=data.status_flag,
    )

    cdf(
        ax,
        data.all_candidates["ttvperiod"],
        include_zero=(x_scale == "linear"),
        **PLOT_FORMAT["ALL"],
    )
    cdf(
        ax,
        data.singles["ttvperiod"],
        include_zero=(x_scale == "linear"),
        **PLOT_FORMAT["SINGLES"],
    )
    cdf(
        ax,
        data.multis["ttvperiod"],
        include_zero=(x_scale == "linear"),
        **PLOT_FORMAT["MULTIS"],
    )
    cdf(
        ax,
        data.m2["ttvperiod"],
        include_zero=(x_scale == "linear"),
        **PLOT_FORMAT["M2"],
    )
    cdf(
        ax,
        data.m3_plus["ttvperiod"],
        include_zero=(x_scale == "linear"),
        **PLOT_FORMAT["M3_PLUS"],
    )
    cdf(
        ax,
        data.innermost_multi["ttvperiod"],
        include_zero=(x_scale == "linear"),
        **PLOT_FORMAT["M_INNERMOST"],
    )
    cdf(
        ax,
        data.middle_multi["ttvperiod"],
        include_zero=(x_scale == "linear"),
        **PLOT_FORMAT["M_MIDDLE"],
    )
    cdf(
        ax,
        data.outermost_multi["ttvperiod"],
        include_zero=(x_scale == "linear"),
        **PLOT_FORMAT["M_OUTERMOST"],
    )

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


def cdf_with_ttv_flag(ax: Axes, data: KeplerData, *, x_scale: str = "log") -> None:
    data = KeplerData(
        data.singles.query("nttobs >= 3"),
        data.multis.query("nttobs >= 3"),
        status_flag=data.status_flag,
    )
    logging.log(
        logging.INFO,
        f"Additional restriction for data without ttvflag filter: nttobs >= 3",
    )

    # Regex search for:
    #    - First digit is 1 OR
    #    - Second digit is 1 or 2 OR
    #    - Third digit is 8 or 9
    # Matches are joined with a binary OR
    # Learn more at: https://regex101.com/r/4KjNjd/1
    ttv_regex = r"t?1\d{2}|t?\d[12]\d|t?\d{2}[89]"
    data_w_ttv = KeplerData(
        data.singles.query("ttvflag.str.match(@ttv_regex)"),
        data.multis.query("ttvflag.str.match(@ttv_regex)"),
        status_flag=data.status_flag,
    )
    data_wo_ttv = KeplerData(
        data.singles.query("~ttvflag.str.match(@ttv_regex)"),
        data.multis.query("~ttvflag.str.match(@ttv_regex)"),
        status_flag=data.status_flag,
    )
    logging.log(
        logging.INFO,
        f"Additional restriction for data with ttvflag filter: "
        f"nttobs >= 3 and ttvflag must match regex {ttv_regex}",
    )

    # Some additional formatting for the plot is required here
    fmt = PLOT_FORMAT["SINGLES"].copy()
    fmt["label"] = f"{fmt['label']} w TTV"
    cdf(ax, data_w_ttv.singles["ttvperiod"], include_zero=(x_scale == "linear"), **fmt)

    fmt["label"] = fmt["label"].replace("w", "w/o")
    fmt["linestyle"] = "dashed"
    cdf(ax, data_wo_ttv.singles["ttvperiod"], include_zero=(x_scale == "linear"), **fmt)

    fmt = PLOT_FORMAT["MULTIS"].copy()
    fmt["label"] = f"{fmt['label']} w TTV"
    cdf(ax, data_w_ttv.multis["ttvperiod"], include_zero=(x_scale == "linear"), **fmt)

    fmt["label"] = fmt["label"].replace("w", "w/o")
    fmt["linestyle"] = "dashed"
    cdf(ax, data_wo_ttv.multis["ttvperiod"], include_zero=(x_scale == "linear"), **fmt)

    ax.set_xscale(x_scale)  # type: ignore
    if x_scale == "linear":
        ax.set_xlim(0, 80)

    ax.set_xlabel("Period [days]")
    ax.set_ylabel("Normalized CDF")
    if x_scale == "log":
        ax.legend(loc="lower right", markerfirst=False)

    save_figure(ax, f"period_cdf_with_ttv_flag_{x_scale}.pdf")


def fraction_of_transiting_companions(
    ax: Axes, data: KeplerData, *, large_planet_cutoff: float = 5.0
) -> None:
    """For each period in our dataset, find the fraction of candidates with a transit companion.

    For instance: for a period of x days how many candidates have a transiting companion that has a smaller/larger orbit?

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot the cdf on.
    data : KeplerData
        The data to plot the cdf of.

    Returns
    -------
    None
    """

    def fraction_of_neighbours(filtered_period, companion_filter) -> float:
        if any(filtered_period):
            return sum(filtered_period & companion_filter) / sum(filtered_period)

        return 0.0

    df = data.all_candidates.sort_values(by="ttvperiod")
    period = df["ttvperiod"]

    # if you are in the middle/outermost position, you have at least one planet with a smaller orbit
    # if you are in the innermost/middle position, you have at least one planet with a larger orbit
    have_inner_companions = df["position"].isin(["middle", "outermost"])
    have_outer_companions = df["position"].isin(["innermost", "middle"])

    large_planets = df["radius"] > large_planet_cutoff
    logging.log(logging.INFO, f"Large planets have R > {large_planet_cutoff} R_earth")

    # find planets that have period smaller/larger than a given period
    # of those planets, find the fraction that also have a transiting companion
    inner_companions = [
        fraction_of_neighbours(period < p, have_inner_companions) for p in period
    ]
    outer_companions = [
        fraction_of_neighbours(period > p, have_outer_companions) for p in period
    ]

    # find large planets that have period smaller/larger than a given period
    # of those planets, find the fraction that also have a transiting companion
    large_inner_companions = [
        fraction_of_neighbours(large_planets & (period < p), have_inner_companions)
        for p in period
    ]
    large_outer_companions = [
        fraction_of_neighbours(large_planets & (period > p), have_outer_companions)
        for p in period
    ]

    # define plot formatting for each line
    plot_properties = (
        (
            inner_companions,
            {
                "color": "#061",
                "linestyle": "solid",
                "label": r"Inner Companions",
            },
        ),
        (
            outer_companions,
            {
                "color": "#f60",
                "linestyle": "solid",
                "label": r"Outer Companions",
            },
        ),
        (
            large_inner_companions,
            {
                "color": "#061",
                "linestyle": "dashed",
                "label": (f"$R_p > {large_planet_cutoff}$ " r"R$_\oplus$ with I.C."),
            },
        ),
        (
            large_outer_companions,
            {
                "color": "#f60",
                "linestyle": "dashed",
                "label": (f"$R_p > {large_planet_cutoff}$ " r"R$_\oplus$ with O.C."),
            },
        ),
    )

    for line_data, plot_format in plot_properties:
        ax.plot(period, line_data, **plot_format)

    ax.set_xscale("log")
    ax.set_xlabel("Period [days]")
    ax.set_ylabel("Cumulative Fraction")
    ax.legend(loc="center left", bbox_to_anchor=(0, 0.6))

    save_figure(ax, "fraction_of_transiting_companions.pdf")
