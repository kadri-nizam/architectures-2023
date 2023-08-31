import pytest
from architectures_2023 import data


@pytest.fixture
def test_data():
    return data.load_data("tests/data/test_set.csv")


@pytest.fixture
def config():
    def _make_config(
        pre_split_filtering: bool,
        demote_multis: bool,
    ) -> dict[str, dict]:
        return {
            "data_processing": {
                "pre_split_filtering": pre_split_filtering,
                "demote_multis_to_singles": demote_multis,
            },
            "data_filtering": {
                "min_ttvperiod": 0,
                "min_snr": 12,
            },
        }

    return _make_config
