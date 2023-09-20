import logging
from functools import partial

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from architectures_2023.data import KeplerData
from architectures_2023.visual import PLOT_FORMAT, cdf, save_figure

__all__ = ["generate_figures", "generate_mono_transit_figures"]


def generate_figures(kepler: KeplerData):
    logging.log(
        logging.INFO, f"Generating figures for radius related data with {kepler}"
    )

    ax = plt.subplot()

    cdf_impact_param_over_radii_subsets(
        ax, kepler, large_planet_cutoff=5, normalize_at_x=1.0
    )

    cdf_population_period_with_radius_subsets(
        ax, kepler, radius_bins=[0, 1.8, 5, 10, 1e12], b_cutoff=0.95
    )


def generate_mono_transit_figures(kepler: KeplerData) -> None:
    logging.log(
        logging.INFO, f"Generating figures for radius related data with {kepler}"
    )

    ax = plt.subplot()

    cdf_radii_of_population_subsets(ax, kepler, b_cutoff=0.95, normalize_at_x=5)

    cdf_impact_param_over_radii_subsets(
        ax, kepler, large_planet_cutoff=5, normalize_at_x=1.0
    )

    long_period_singles = partial(
        cdf_radii_of_long_period_singles,
        ax,
        kepler,
        b_cutoff=0.95,
        long_period_cutoff=10,
        normalize_at_x=10,
        x_scale="linear",
    )
    long_period_singles(large_planet_cutoff=3)
    long_period_singles(large_planet_cutoff=4)
    long_period_singles(large_planet_cutoff=4.5)
    long_period_singles(large_planet_cutoff=5)


def cdf_impact_param_over_radii_subsets(
    ax: Axes,
    data: KeplerData,
    *,
    large_planet_cutoff: float = 5.0,
    normalize_at_x: float = 1.0,
) -> None:
    radius_cutoff_label = rf"$R_p < {large_planet_cutoff}$ " r"R$_\oplus$"

    plot_cdf = partial(cdf, ax, normalize_at_x=normalize_at_x)

    fmt = PLOT_FORMAT["SINGLES"].copy()
    fmt["label"] = f"{fmt['label']}: {radius_cutoff_label}"
    plot_cdf(data.singles.query("radius < @large_planet_cutoff")["b"], **fmt)

    fmt["label"] = fmt["label"].replace("<", r"\geq")
    fmt["linestyle"] = "dashed"
    plot_cdf(data.singles.query("radius >= @large_planet_cutoff")["b"], **fmt)

    fmt = PLOT_FORMAT["MULTIS"].copy()
    fmt["label"] = rf"{fmt['label']}: {radius_cutoff_label}"
    plot_cdf(data.multis.query("radius < @large_planet_cutoff")["b"], **fmt)

    fmt["label"] = fmt["label"].replace("<", r"\geq")
    fmt["linestyle"] = "dashed"
    plot_cdf(data.multis.query("radius >= @large_planet_cutoff")["b"], **fmt)

    ax.set_xlabel(r"Impact Parameter [$b$]")
    ax.set_ylabel(f"CDF normalized at $b = {normalize_at_x}$")
    ax.legend(loc="lower right", markerfirst=False)

    save_figure(ax, f"radius_w_mono_cdf_impact_param_over_radii_subsets.pdf")


def cdf_radii_of_population_subsets(
    ax: Axes,
    data: KeplerData,
    *,
    b_cutoff: float = 0.95,
    normalize_at_x: float = 1.0,
    x_scale: str = "log",
) -> None:
    logging.log(logging.INFO, f"Additional restriction: b + b_ep < {b_cutoff}")
    df = KeplerData(
        data.singles.query("b + b_ep < @b_cutoff"),
        data.multis.query("b + b_ep < @b_cutoff"),
        status_flag=data.status_flag,
    )

    plot_group = (
        {"SINGLES": df.singles["radius"], "MULTIS": df.multis["radius"]},
        {"M2": df.m2["radius"], "M3_PLUS": df.m3_plus["radius"]},
    )

    for group in plot_group:
        for k, v in group.items():
            cdf(
                ax,
                v,
                normalize_at_x=normalize_at_x,
                **PLOT_FORMAT[k],
            )

        ax.set_xscale(x_scale)  # type: ignore
        ax.set_xlabel(r"Radius [$\mathrm{R}_\oplus$]")
        ax.set_ylabel(f"CDF normalized at $R_p = {normalize_at_x}$ " r"R$_\oplus$")
        ax.legend(loc="lower right", markerfirst=False)
        ax.set_ylim(-0.01, 1.21)

        save_figure(
            ax, f"radius_w_mono_cdf_{'_'.join(group.keys()).lower()}_{x_scale}.pdf"
        )


def cdf_radii_of_long_period_singles(
    ax: Axes,
    data: KeplerData,
    *,
    normalize_at_x: float = 1.0,
    x_scale: str = "log",
    b_cutoff: float = 0.95,
    large_planet_cutoff: float = 5.0,
    long_period_cutoff: float = 10.0,
) -> None:
    logging.log(
        logging.INFO,
        f"Additional restriction: b + b_ep < {b_cutoff} and radius > {large_planet_cutoff}",
    )
    df = KeplerData(
        data.singles.query("(b + b_ep < @b_cutoff) & (radius > @large_planet_cutoff)"),
        data.multis.query("(b + b_ep < @b_cutoff) & (radius > @large_planet_cutoff)"),
        status_flag=data.status_flag,
    )

    cdf(
        ax,
        df.singles["radius"],
        normalize_at_x=normalize_at_x,
        start_cdf_at=large_planet_cutoff,
        **PLOT_FORMAT["SINGLES"],
    )
    cdf(
        ax,
        df.multis["radius"],
        normalize_at_x=normalize_at_x,
        start_cdf_at=large_planet_cutoff,
        **PLOT_FORMAT["MULTIS"],
    )

    logging.log(
        logging.INFO, f"Long period planets have period > {long_period_cutoff} days"
    )
    fmt = PLOT_FORMAT["SINGLES"].copy()
    fmt["label"] = rf"{fmt['label']} $(|P| \geq {long_period_cutoff}$ days$)$"
    fmt["linewidth"] = 1.0
    fmt["linestyle"] = "dashed"
    cdf(
        ax,
        df.singles.query("abs(ttvperiod) > @long_period_cutoff")["radius"],
        normalize_at_x=normalize_at_x,
        start_cdf_at=large_planet_cutoff,
        **fmt,
    )

    ax.set_xscale(x_scale)  # type: ignore
    if x_scale == "linear":
        ax.set_xlim(large_planet_cutoff - 1, 25)

    ax.set_xlabel(r"Radius [$\mathrm{R}_\oplus$]")
    ax.set_ylabel(f"CDF normalized at $R_p = {normalize_at_x}$ " r"R$_\oplus$")
    ax.legend(loc="lower right", markerfirst=False)

    save_figure(
        ax,
        "radius_w_mono_cdf_long_period_singles_gt_"
        f"{str(large_planet_cutoff).replace('.', '_')}"
        f"REarth_{x_scale}.pdf",
    )


def cdf_population_period_with_radius_subsets(
    ax: Axes,
    data: KeplerData,
    *,
    radius_bins: list[float],
    b_cutoff: float = 0.95,
    x_scale: str = "log",
) -> None:
    logging.log(logging.INFO, f"Additional restriction: b + b_ep < {b_cutoff}")
    df = KeplerData(
        data.singles.query("b + b_ep < @b_cutoff"),
        data.multis.query("b + b_ep < @b_cutoff"),
        status_flag=data.status_flag,
    )

    # get the bin indices (based on radius) for each planet
    singles_bins = pd.cut(
        df.singles["radius"], bins=radius_bins, include_lowest=True, labels=False
    )
    multis_bins = pd.cut(
        df.multis["radius"], bins=radius_bins, include_lowest=True, labels=False
    )

    # this plot requires special formatting for the legend and the lines which
    # the label formatting will be taken care of in the for loop
    temp_label = r"{small} R$_\oplus < R_p \leq$ {large} R$_\oplus$"
    plot_formats = [
        {"label": r"$R_p \leq$ {large} R$_\oplus$", "linewidth": 1},
        {"linestyle": "dashed", "linewidth": 1, "label": temp_label},
        {"linestyle": "dotted", "linewidth": 1, "label": temp_label},
        {"linestyle": "dashdot", "linewidth": 1, "label": r"{small} R$_\oplus < R_p$"},
    ]

    for bin_idx, fmt in zip(range(len(radius_bins) - 1), plot_formats):
        # additional label formatting
        fmt["label"] = (
            r"[{}] "
            f"{fmt['label'].format(small=radius_bins[bin_idx], large=radius_bins[bin_idx + 1])}"
        )

        cdf(
            ax,
            df.singles[singles_bins == bin_idx]["ttvperiod"],
            **PLOT_FORMAT["SINGLES"] | fmt,
        )
        cdf(
            ax,
            df.multis[multis_bins == bin_idx]["ttvperiod"],
            **PLOT_FORMAT["MULTIS"] | fmt,
        )

    ax.set_xscale(x_scale)  # type: ignore
    ax.set_xlim(0.08, 8_000)
    ax.set_xlabel("Period [days]")
    ax.set_ylabel("Normalized CDF")

    # we'll have to break the legend into two sections because there are a lot of lines
    # See: https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html#multiple-legends-on-the-same-axes
    handles, labels = ax.get_legend_handles_labels()
    # even indices are singles, odd indices are multis
    upper_left = ax.legend(
        handles[::2],
        labels[::2],
        loc="upper left",
        handletextpad=0.5,
    )
    ax.legend(
        handles[1::2],
        labels[1::2],
        loc="lower right",
        markerfirst=False,
        handletextpad=0.5,
    )
    ax.add_artist(upper_left)

    save_figure(ax, f"radius_cdf_population_period_with_radius_subsets_{x_scale}.pdf")
