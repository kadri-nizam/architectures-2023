import logging
from datetime import datetime as dt

from architectures_2023 import data, period, radius, snr

# logging setup
log_file = f"{dt.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    filename=log_file,
    encoding="utf-8",
    format="%(levelname)s => %(asctime)s => %(funcName)s => %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)


def main():
    df, _ = data.load_data()
    config = data.load_config()

    logging.log(logging.INFO, "Data loaded and processed.")
    kepler_p = data.process_data(df, config, data.STATUS_FLAG.PERIOD_RELATED)
    period.generate_figures(kepler_p)

    kepler_r = data.process_data(df, config, data.STATUS_FLAG.RADIUS_RELATED)
    radius.generate_figures(kepler_r)

    # some figures in the paper include single transit planets with negative periods
    # those are done here
    config["data_filtering"]["min_ttvperiod"] = None
    logging.log(logging.INFO, "Config now allows negative period PCs.")

    kepler_monos_included_r = data.process_data(
        df, config, data.STATUS_FLAG.RADIUS_RELATED
    )
    radius.generate_mono_transit_figures(kepler_monos_included_r)

    config["data_filtering"]["min_snr"] = None
    logging.log(logging.INFO, "Config now allows negative period PCs and all SNR.")
    kepler_monos_included_p = data.process_data(
        df, config, data.STATUS_FLAG.PERIOD_RELATED
    )
    snr.generate_mono_transit_figures(kepler_monos_included_p)

    logging.log(logging.INFO, f"{10 * '='} END OF RUN {10 * '='}")


if __name__ == "__main__":
    main()
