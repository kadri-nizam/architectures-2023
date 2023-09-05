import logging
from datetime import datetime as dt
from math import log

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
    kepler_r = data.process_data(df, config, data.STATUS_FLAG.RADIUS_RELATED)

    period.generate_figures(kepler_p)
    radius.generate_figures(kepler_r)


if __name__ == "__main__":
    main()
