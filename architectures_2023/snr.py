import logging

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from architectures_2023.data import KeplerData
from architectures_2023.visual import PLOT_FORMAT, save_figure

__all__ = ["generate_mono_transit_figures"]


def generate_mono_transit_figures(kepler: KeplerData):
    ax = plt.subplot()

    logging.log(
        logging.INFO, f"Generating figures for SNR related data with:\n{kepler}"
    )

    counts_of_snr(ax, kepler, x_scale="log")
    counts_of_kepler_mag(ax, kepler, y_scale="log")


def counts_of_snr(ax: Axes, data: KeplerData, *, x_scale: str = "log") -> None:
    bins = 10 ** np.arange(
        np.log10(int(data.all_candidates["snr"].min())),
        np.log10(int(data.all_candidates["snr"].max())),
        0.029,
        # 0.032,
        # 0.045,
    )

    ax.hist(
        data.singles["snr"],
        bins=bins,
        histtype="step",
        **PLOT_FORMAT["SINGLES"] | {"label": "Singles"},
    )
    ax.hist(
        data.multis["snr"],
        bins=bins,
        histtype="step",
        **PLOT_FORMAT["MULTIS"] | {"label": "Multis"},
    )

    fmt = PLOT_FORMAT["MULTIS"].copy()
    fmt["linewidth"] = 1
    fmt["linestyle"] = "dashed"
    fmt["label"] = f"Weakest S/N in Multis"
    ax.hist(
        data.multis.groupby("system")["snr"].min(),
        bins=bins,
        histtype="step",
        **fmt,
    )

    ax.set_xscale(x_scale)  # type: ignore
    ax.set_xlabel("S/N")
    ax.set_ylabel("Number of Planet Candidates")
    ax.set_xlim(5, 1000)

    ax.legend(loc="upper right", markerfirst=False)
    save_figure(ax, f"snr_hist_counts_{x_scale}.pdf")


def counts_of_kepler_mag(ax: Axes, data: KeplerData, *, y_scale: str = "log") -> None:
    bins = np.arange(6.5, 18, 0.5)

    ax.hist(
        data.singles["kepmag"],
        bins=bins,
        histtype="step",
        **PLOT_FORMAT["SINGLES"] | {"label": "Singles"},
    )

    fmt = PLOT_FORMAT["MULTIS"].copy()
    ax.hist(
        data.multis.groupby("system")["kepmag"].first(),
        bins=bins,
        histtype="step",
        **fmt | {"label": f"Systems of Multis"},
    )

    fmt["linewidth"] = 1
    fmt["linestyle"] = "dashed"
    fmt["label"] = f"PCs in Multis"
    ax.hist(data.multis["kepmag"], bins=bins, histtype="step", **fmt)

    ax.set_yscale(y_scale)  # type: ignore
    ax.set_xlabel(r"\textit{Kepler} Magnitude of Host Star")
    ax.set_ylabel("Number of Planet Candidates")

    ax.set_xlim(left=7.5)

    ax.legend(loc="upper left")
    save_figure(ax, f"snr_hist_kepler_mag_counts.pdf")
