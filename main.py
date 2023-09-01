import logging
from datetime import datetime as dt

from architectures_2023 import data, period

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

    kepler = data.process_data(df, config, data.STATUS_FLAG.PERIOD_RELATED)
    logging.log(logging.INFO, f"Data loaded and processed.\n\n{kepler}")

    period.generate_figures(kepler)


if __name__ == "__main__":
    main()
