import pytest
from heart_disease.data import clean_data, download_raw


@pytest.fixture(scope="module")
def raw_file(tmp_path_factory):
    path = download_raw(tmp_path_factory.mktemp("raw"))
    return path


def test_clean_data_no_missing_targets(raw_file):
    df = clean_data(raw_file, save=False)
    assert "target" in df.columns
    assert df["target"].isin([0, 1]).all()
    assert df.isna().sum().sum() == 0


def test_processed_schema(raw_file):
    df = clean_data(raw_file, save=False)
    assert len(df.columns) == 14  # 13 features + target
    assert set(df["target"].unique()).issubset({0, 1})
