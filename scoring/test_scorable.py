"""Module to check that your flood tool will be scorable."""

import flood_tool
import pandas as pd
import numbers

tool = flood_tool.Tool()
POSTCODES = ["BA1 5NB", "RH16 2QE"]
EASTINGS = [374999, 535295]
NORTHINGS = [165441, 123643]

PRICE_METHODS = flood_tool.house_price_models
CLASS_METHODS = flood_tool.flood_class_from_postcode_models
LCLASS_METHODS = flood_tool.flood_class_from_location_models
LAUTH_METHODS = flood_tool.local_authority_models
HIST_METHODS = flood_tool.historic_flooding_models

for method in (
    list(PRICE_METHODS.keys())
    + list(CLASS_METHODS.keys())
    + list(LCLASS_METHODS.keys())
    + list(LAUTH_METHODS.keys())
    + list(HIST_METHODS.keys())
):
    tool.fit_to_data(models=[method])


def test_lookup_lat_long():
    """Check return type."""
    data = tool.lookup_lat_long(POSTCODES)
    assert isinstance(data, pd.DataFrame)
    for postcode in POSTCODES:
        assert postcode in data.index
    assert 'latitude' in data
    assert 'longitude' in data


def test_predict_flood_class_from_postcode():
    """Check return types."""
    for method in CLASS_METHODS.keys():
        data = tool.predict_flood_class_from_postcode(POSTCODES, method)
        assert isinstance(data, pd.Series)
        for postcode in POSTCODES:
            assert postcode in data.index
        assert all([isinstance(val, numbers.Number) for val in data.values])


def test_predict_flood_class_from_OSGB36_location():
    """Check return types."""
    for method in LCLASS_METHODS.keys():
        data = tool.predict_flood_class_from_OSGB36_location(
            EASTINGS, NORTHINGS, method
        )
        assert isinstance(data, pd.Series)
        for east, north in zip(EASTINGS, NORTHINGS):
            assert (east, north) in data.index
        assert all([isinstance(val, numbers.Number) for val in data.values])


def test_predict_median_house_price():
    """Check return type."""
    for method in PRICE_METHODS.keys():
        data = tool.predict_median_house_price(POSTCODES, method)
        assert isinstance(data, pd.Series)
        for postcode in POSTCODES:
            assert postcode in data.index
        assert all([isinstance(val, numbers.Number) for val in data.values])


def test_predict_local_authority():
    """Check return type."""
    for method in LAUTH_METHODS.keys():
        data = tool.predict_local_authority(EASTINGS, NORTHINGS, method)
        assert isinstance(data, pd.Series)
        for east, north in zip(EASTINGS, NORTHINGS):
            assert (east, north) in data.index


def test_predict_historic_flooding():
    """Check return type."""
    for method in HIST_METHODS.keys():
        data = tool.predict_historic_flooding(POSTCODES, method)
        assert isinstance(data, pd.Series)
        for postcode in POSTCODES:
            assert postcode in data.index
