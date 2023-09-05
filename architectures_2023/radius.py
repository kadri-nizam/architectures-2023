import logging
from functools import partial

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from architectures_2023.data import KeplerData
from architectures_2023.visual import PLOT_FORMAT, cdf, save_figure


def generate_figures(kepler: KeplerData):
    logging.log(
        logging.INFO, f"Generating figures for radius related data with {kepler}"
    )

    ax = plt.subplot()

    cdf_impact_param_over_radii_subsets(
        ax, kepler, large_planet_cutoff=2.5, normalize_at_x=1.0
    )

    cdf_radii_of_population_subsets(ax, kepler, b_cutoff=0.95, normalize_at_x=5)

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

    save_figure(ax, f"radius_cdf_impact_param_over_radii_subsets.pdf")


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

        save_figure(ax, f"radius_cdf_{'_'.join(group.keys()).lower()}_{x_scale}.pdf")


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
    fmt["label"] = rf"{fmt['label']} $(P \geq {long_period_cutoff}$ days$)$"
    fmt["linewidth"] = 1.0
    fmt["linestyle"] = "dashed"
    cdf(
        ax,
        df.singles.query("ttvperiod > @long_period_cutoff")["radius"],
        normalize_at_x=normalize_at_x,
        start_cdf_at=large_planet_cutoff,
        **fmt,
    )

    ax.set_xscale(x_scale)  # type: ignore
    if x_scale == "linear":
        ax.set_xlim(large_planet_cutoff - 1, 25)

    # additional info on what we define as "large" planets
    ax.text(
        0.05,
        0.94,
        rf"Subset of PCs with $R_p > {large_planet_cutoff}$ R$_\oplus$",
        fontsize=13,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )

    ax.set_xlabel(r"Radius [$\mathrm{R}_\oplus$]")
    ax.set_ylabel(f"CDF normalized at $R_p = {normalize_at_x}$ " r"R$_\oplus$")
    ax.legend(loc="lower right", markerfirst=False)

    save_figure(
        ax,
        f"radius_cdf_long_period_singles_gt_{large_planet_cutoff}REarth_{x_scale}.pdf",
    )
