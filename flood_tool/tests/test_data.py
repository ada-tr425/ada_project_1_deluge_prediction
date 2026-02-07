"""Test default data compatibility with flood tool."""

import flood_tool

import pandas as pd


def test_unlabelled_data():

    RAW_FEATURES = flood_tool.tool.RAW_FEATURES
    DEFAULT_FEATURE_DATA = flood_tool.tool.DEFAULT_FEATURE_DATA

    assert len(DEFAULT_FEATURE_DATA.index) > 0
    assert DEFAULT_FEATURE_DATA.columns.isin(RAW_FEATURES).all()
    assert DEFAULT_FEATURE_DATA.duplicated(subset="postcode").sum() == 0
    assert pd.Series(RAW_FEATURES).isin(DEFAULT_FEATURE_DATA.columns).all()


def test_labelled_data():

    RAW_FEATURES = flood_tool.tool.RAW_FEATURES
    DEFAULT_UNIT_DATA = flood_tool.tool.DEFAULT_UNIT_DATA

    assert len(DEFAULT_UNIT_DATA.index) > 0
    assert pd.Series(RAW_FEATURES).isin(DEFAULT_UNIT_DATA.columns).all()
    assert DEFAULT_UNIT_DATA.duplicated(subset="postcode").sum() == 0
