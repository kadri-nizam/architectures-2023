import logging
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Any, Callable

import pandas as pd


class STATUS_FLAG(StrEnum):
    """Enum for the status flag to filter the data during preprocessing."""

    PERIOD_RELATED = r"^P.*"
    RADIUS_RELATED = r"^[PR].*"
    R_DISPOSITION_RELATED = r"^R.*"


@dataclass(frozen=True)
class KeplerData:
    """Class to hold the Kepler data and the configurations used for processing the raw data. Helper functions are provided for easy access to the data."""

    singles: pd.DataFrame
    multis: pd.DataFrame
    config: dict[str, Any] | None = None
    status_flag: STATUS_FLAG | None = None

    def __repr__(self):
        info = f"KeplerData(singles:{len(self.singles)}, multis:{len(self.multis)})\n"
        info += f"   Status Flag: {self.status_flag}\n"
        info += f"   M2: {len(self.m2)}\n"
        info += f"   M3+: {len(self.m3_plus)}\n"

        return info

    @property
    def all_candidates(self):
        return (
            pd.concat([self.singles, self.multis])
            .reset_index(drop=True)
            .sort_values(by=["system", "ttvperiod"])
        )

    @property
    def m2(self) -> pd.DataFrame:
        return self.get_multis_system_with(num_planets=2)

    @property
    def m3(self) -> pd.DataFrame:
        return self.get_multis_system_with(num_planets=3)

    @property
    def m3_plus(self) -> pd.DataFrame:
        return self.get_multis_system_with(num_planets=3, operator=">=")

    @property
    def innermost_multi(self) -> pd.DataFrame:
        return self.multis.query("position == 'innermost'")

    @property
    def outermost_multi(self) -> pd.DataFrame:
        return self.multis.query("position == 'outermost'")

    @property
    def middle_multi(self) -> pd.DataFrame:
        return self.multis.query("position == 'middle'")

    def get_multis_system_with(
        self, *, num_planets: int, operator: str = "=="
    ) -> pd.DataFrame:
        return self.multis.query(f"multiplicity {operator} {num_planets}")


def load_config(config_file_path: str = "") -> dict[str, Any]:
    """Loads a configuration file specifying details on data import/processing and analysis.

    Parameters:
    -----------
        data_path : str, optional
            Path to the configuration file. If no path is specified, the default config.toml file in the root directory is loaded.

    Returns:
    --------
        dict[str, Any]
            Configuration dictionary.
    """
    import json

    if not config_file_path:
        config_file_path = f"{Path(__file__).parent / '../config.toml'}"

    with open(config_file_path, "rb") as f:
        config = tomllib.load(f)
        logging.log(logging.INFO, f"Config:\n{json.dumps(config, indent=3)}")
        return config


def load_data(data_path: str = "") -> tuple[pd.DataFrame, str]:
    """Reads the latest CSV file and cleans the dataframe for analysis.

    Parameters:
    -----------
        data_path : str, optional
            Path to the CSV file. Loads the file with the latest date in its name from data/raw if no path is specified.

    Returns:
    --------
        tuple[pd.DataFrame, str]
            Dataframe and name of the file loaded.
    """
    root = Path(__file__).parent / ".."

    if not data_path:
        path = root / "data" / "raw"

        # Glob returns a list of paths with the CSV extension
        # max returns the latest file
        data_path = str(max(path.glob("*.csv")))

    if (processed := root / "data" / "processed" / Path(data_path).name).exists():
        logging.log(logging.INFO, f"Cached data found. Loading from {processed}")
        df = pd.read_csv(processed)
        return df, str(processed)

    logging.log(logging.INFO, f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df = clean_data(df)

    # cache the processed dataframe for future use
    logging.log(logging.INFO, f"Caching processed data to {processed}")
    df.to_csv(processed, index=False)

    return df, data_path


def process_data(
    df: pd.DataFrame,
    config: dict,
    status_flag: STATUS_FLAG,
) -> KeplerData:
    """Prepare data for analysis.

    Data is filtered, split into multi-planet and single-planet systems and demoted if necessary. Configuration is read from the config.toml file.

    Parameters:
    -----------
        df : pd.DataFrame
            Dataframe to process.
        config : dict[str, Any]
            Configuration dictionary based on the config.toml file.
        status_flag : STATUS_FLAG, optional
            Status flag to filter the data. If not provided, all data is used.

    Returns:
    --------
        KeplerData
            Instance of KeplerData containing the singles and multis dataframes.
    """

    fn = partial(filter_data, status_flag=status_flag, **config["data_filtering"])

    if config["data_processing"]["pre_split_filtering"]:
        singles, multis = _filter_then_split(df, fn)
    else:
        singles, multis = _split_then_filter(df, fn)

    singles = _label_position(singles)
    multis = _label_position(multis)

    if config["data_processing"]["demote_multis_to_singles"]:
        singles, multis = demote_multis_to_singles(singles, multis)

    return KeplerData(singles, multis, config, status_flag)


def filter_data(
    df: pd.DataFrame,
    status_flag: STATUS_FLAG | None = None,
    min_ttvperiod: float = 0,
    min_snr: float = 12,
) -> pd.DataFrame:
    """Filters the dataframe based on the status flag, minimum TTV period and minimum SNR.

    status_flag can be one of the following:
    - "^P.*" for all period related statistics
    - "^[PR].*" for all radius related statistics
    - "^R.*" for all radius-disposition related statistics

    Parameters:
    -----------
        df : pd.DataFrame
            Dataframe to filter.

    Returns:
    --------
        pd.DataFrame
            Filtered dataframe.
    """
    if status_flag is not None:
        df = df[df["statusflag"].str.match(status_flag)]

    df = df[df["ttvperiod"] >= min_ttvperiod]
    df = df[df["snr"] >= min_snr]

    return df.reset_index(drop=True)


def get_multis_and_singles(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divides the full Kepler dataframe into single-planet and multi-planet systems.

    Parameters:
    -----------
        df : pd.DataFrame
            Full Kepler dataset.

    Returns:
    --------
        tuple[pd.DataFrame, pd.DataFrame]
            Dataframes with single-planet and multi-planet systems.
    """

    multis_mask = df["system"].duplicated(keep=False)
    multis = df[multis_mask].reset_index(drop=True)

    singles = df.drop(df[multis_mask].index).reset_index(drop=True)

    return singles, multis


def demote_multis_to_singles(
    singles: pd.DataFrame,
    multis: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Demote multi-planet systems to single-planet systems if the system has been filtered down to a single planet.

    Parameters:
    -----------
        singles : pd.DataFrame
            Single-planet systems.
        multis : pd.DataFrame
            Multi-planet systems.

    Returns:
    --------
        tuple[pd.DataFrame, pd.DataFrame]
            Updated single-planet and multi-planet systems.
    """

    new_singles, multis = get_multis_and_singles(multis)

    new_singles["position"] = "demoted"
    singles = (
        pd.concat([singles, new_singles])
        .sort_values(by=["system"])
        .reset_index(drop=True)
    )
    singles["position"] = singles["position"].astype("category")

    return singles, multis


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Converts pandas dataframe columns to lowercase, replaces spaces with underscores and removes special characters.

    Parameters:
    -----------
        df : pd.DataFrame
            Dataframe to clean.

    Returns:
    --------
        pd.DataFrame
            Dataframe with column names cleaned up.
    """
    df.columns = (
        df.columns.str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.replace("[", "")
        .str.replace("]", "")
        .str.replace(":", "")
        .str.replace("?", "")
        .str.replace("'", "")
        .str.replace("/", "_")
        .str.replace("-", "_")
        .str.replace(".", "_")
        .str.replace(",", "_")
        .str.replace("&", "_")
        .str.replace("__", "_")
        .str.replace("__", "_")
    )

    return df


def _filter_then_split(
    df: pd.DataFrame,
    filterer: Callable[[pd.DataFrame], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = filterer(df)
    df["multiplicity"] = df.groupby("system")["koi"].transform("count")
    return get_multis_and_singles(df)


def _split_then_filter(
    df: pd.DataFrame,
    filterer: Callable[[pd.DataFrame], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df["multiplicity"] = df.groupby("system")["koi"].transform("count")
    singles, multis = map(filterer, get_multis_and_singles(df))
    return singles, multis


def _label_position(df: pd.DataFrame) -> pd.DataFrame:
    """Labels the planets in the dataframe as innermost, outermost or middle. Single-planet systems are labeled as single.

    Parameters:
    -----------
        df : pd.DataFrame
            Dataframe to label.

    Returns:
    --------
        pd.DataFrame
            Labeled dataframe.
    """

    # Label all planets as middle first
    df["position"] = "middle"

    # Then label the innermost and outermost planets
    df.loc[df.groupby("system")["ttvperiod"].idxmin(), "position"] = "innermost"
    df.loc[df.groupby("system")["ttvperiod"].idxmax(), "position"] = "outermost"

    # Single-planet systems are labeled as single
    df.loc[~df["system"].duplicated(keep=False), "position"] = "single"

    # Change to categorical type
    df["position"] = df["position"].astype("category")

    return df


def _convert_column_datatypes(df: pd.DataFrame) -> pd.DataFrame:
    """Converts the column data types to the ones specified in COLUMN_DATATYPE.

    Parameters:
    -----------
        df : pd.DataFrame
            Dataframe to convert.

    Returns:
    --------
        pd.DataFrame
            Converted dataframe.
    """
    for column, datatype in COLUMN_DATATYPE.items():
        df[column] = df[column].astype(datatype)  # type: ignore

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the column names and converts the column data types of the dataframe. Replace NaN values with 0 for the columns "chisqwttv" and "kepmag", and "" for the column "kepler_id".

    Parameters:
    -----------
        df : pd.DataFrame
            Dataframe to clean and convert.

    Returns:
    --------
        pd.DataFrame
            Cleaned and converted dataframe.
    """
    df = _clean_column_names(df)

    nan_column_fix = {"kepler_id": "", "chisqwttv": 0, "kepmag": 0}
    df = df.fillna(nan_column_fix)

    df = _convert_column_datatypes(df)
    df["system"] = df["koi"].astype("float32").astype("int32")

    df = df.sort_values(by=["system", "ttvperiod"]).reset_index(drop=True)

    return df


# Column datatypes for the dataframe
COLUMN_DATATYPE = {
    "kic": "category",
    "koi": "str",
    "kepler_id": "category",
    "ttvperiod": "float64",
    "ttvperiod_e": "float64",
    "epoch": "float64",
    "epoch_e": "float64",
    "rprs": "float64",
    "rprs_p": "float64",
    "rprs_m": "float64",
    "b": "float64",
    "b_ep": "float64",
    "b_em": "float64",
    "rhostar_model": "float64",
    "rhostar_model_ep": "float64",
    "rhostar_model_em": "float64",
    "u1": "float64",
    "u2": "float64",
    "nttobs": "int64",
    "ntt": "int64",
    "ttvflag": "category",
    "tdepth": "float64",
    "tdepth_e": "float64",
    "tdur": "float64",
    "tdur_e": "float64",
    "tduravg": "float64",
    "tduravg_e": "float64",
    "radius": "float64",
    "radius_ep": "float64",
    "radius_em": "float64",
    "snr": "float64",
    "snrwottv_snrwttv": "float64",
    "mes": "float64",
    "chisqwttv": "float64",
    "chisqwottv": "float64",
    "dchisq_inj": "float64",
    "adrs": "float64",
    "adrs_ep": "float64",
    "adrs_em": "float64",
    "incl": "float64",
    "incl_ep": "float64",
    "incl_em": "float64",
    "s0": "float64",
    "s0_ep": "float64",
    "s0_em": "float64",
    "kepmag": "float64",
    "stellar_source": "int8",
    "rhostar": "float64",
    "rhostar_ep": "float64",
    "rhostar_em": "float64",
    "teff": "float64",
    "teff_e": "float64",
    "rstar": "float64",
    "rstar_ep": "float64",
    "rstar_em": "float64",
    "mstar": "float64",
    "mstar_ep": "float64",
    "mstar_em": "float64",
    "logg": "float64",
    "logg_ep": "float64",
    "logg_em": "float64",
    "m_h": "float64",
    "m_h_e": "float64",
    "statusflag": "category",
    "rhostar_model_lc": "float64",
    "epoch_lc": "float64",
    "per_lc": "float64",
    "b_lc": "float64",
    "rprs_lc": "float64",
    "lcflag": "bool",
    "ttvper_cor": "float64",
    "ttvper_cor_err": "float64",
}
