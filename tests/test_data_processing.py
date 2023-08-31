from architectures_2023 import data


def test_multis_singles_split(test_data):
    df, _ = test_data
    singles, multis = data.get_multis_and_singles(df)

    # duplicates in the original dataset should be in multis
    duplicates = df["system"].duplicated(keep=False)

    assert duplicates.sum() == len(multis)
    assert (~duplicates).sum() == len(singles)


def test_filter_data(test_data):
    df, _ = test_data
    df = data.filter_data(df, min_ttvperiod=50, min_snr=20)

    assert not (df["ttvperiod"] <= 50).any()
    assert not (df["snr"] <= 20).any()


def test_demote_multis(test_data):
    df, _ = test_data

    multis, singles = data.get_multis_and_singles(df)
    multis = data.filter_data(multis, data.STATUS_FLAG.PERIOD_RELATED)

    # planets in multis that are the only planet in their system should be demoted
    to_demote = multis[~multis["system"].duplicated(keep=False)]

    _, singles = data.demote_multis_to_singles(multis, singles)

    assert singles.query("position == 'demoted'")["koi"].isin(to_demote["koi"]).all()


def test_process_data(test_data, config):
    df, _ = test_data
    config = config(pre_split_filtering=False, demote_multis=True)

    kepler = data.process_data(df, config, data.STATUS_FLAG.PERIOD_RELATED)

    # ensure positions are processed correctly
    assert not set(kepler.singles["position"]).intersection(
        ["innermost", "outermost", "middle"]
    )
    assert not (kepler.multis["position"] == "single").any()

    # ensure singles are singles and multis are multis
    assert not kepler.singles.duplicated(keep=False).any()
    assert kepler.get_multis_system_with(num_planets=1).empty
