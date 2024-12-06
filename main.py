import logging
from datetime import datetime as dt

import pandas as pd

from architectures_2023 import data, period, radius, snr
from joblib import Parallel, delayed

# logging setup
log_file = f"{dt.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    filename=log_file,
    encoding="utf-8",
    format="%(levelname)s => %(asctime)s => %(funcName)s => %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)


def period_related(df: pd.DataFrame):
    logging.log(logging.INFO, "Loading and processing data for period-related figures.")
    config = data.load_config()
    kepler_p = data.process_data(df, config, data.STATUS_FLAG.PERIOD_RELATED)
    period.generate_figures(kepler_p)


def radius_related(df: pd.DataFrame):
    logging.log(logging.INFO, "Loading and processing data for radius-related figures.")
    config = data.load_config()
    kepler_r = data.process_data(df, config, data.STATUS_FLAG.RADIUS_RELATED)
    radius.generate_figures(kepler_r)


def mono_inclusive_radius_related(df: pd.DataFrame):
    logging.log(
        logging.INFO,
        "Loading and processing data for mono-inclusive, radius-related figures.",
    )
    config = data.load_config()
    logging.log(logging.INFO, "Config now allows negative period PCs.")
    config["data_filtering"]["min_ttvperiod"] = None
    kepler_monos_included_r = data.process_data(
        df, config, data.STATUS_FLAG.RADIUS_RELATED
    )
    radius.generate_mono_transit_figures(kepler_monos_included_r)


def snr_related(df: pd.DataFrame):
    logging.log(
        logging.INFO,
        "Loading and processing data for mono-inclusive, snr-related figures.",
    )
    config = data.load_config()
    config["data_filtering"]["min_ttvperiod"] = None
    config["data_filtering"]["min_snr"] = None
    logging.log(logging.INFO, "Config now allows negative period PCs and all SNR.")
    kepler_monos_included_p = data.process_data(
        df, config, data.STATUS_FLAG.PERIOD_RELATED
    )
    snr.generate_mono_transit_figures(kepler_monos_included_p)


def main():
    df, _ = data.load_data()

    task = []
    task.append(delayed(period_related)(df))
    task.append(delayed(radius_related)(df))

    # include negative period PCs
    task.append(delayed(mono_inclusive_radius_related)(df))

    # include negative period PCs and all SNR
    task.append(delayed(snr_related)(df))

    # Actually execute all the functions in parallel now
    Parallel(n_jobs=4, verbose=20)(task)
    logging.log(logging.INFO, f"{10 * '='} END OF RUN {10 * '='}")


if __name__ == "__main__":
    main()
