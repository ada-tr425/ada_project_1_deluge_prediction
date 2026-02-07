"""Test flood tool."""

import numpy as np

from pytest import mark

import flood_tool.tool as tool


test_tool = tool.Tool()

TEST_DATA = tool.DEFAULT_FEATURE_DATA.iloc[:100, :]


def test_lookup_easting_northing():
    """Check easting/northing lookup."""

    data = test_tool.lookup_easting_northing(["PO15 7QG"])

    assert len(data.index) == 1
    assert "PO15 7QG" in data.index

    assert np.isclose(data.loc["PO15 7QG", "easting"], 452911).all()
    assert np.isclose(data.loc["PO15 7QG", "northing"], 110572).all()


@mark.xfail  # We expect this test to fail until we write some code for it.
def test_lookup_lat_long():
    """Check latitude/longitude lookup."""

    data = test_tool.lookup_lat_long(["PO15 7QG"])

    assert len(data.index) == 1
    assert "PO15 7QG" in data.index

    assert np.isclose(
        data.loc["PO15 7QG", "latitude"], 50.7863, rtol=1.0e-3
    ).all()
    assert np.isclose(
        data.loc["PO15 7QG", "longitude"], -7.0257, rtol=1.0e-3
    ).all()


def test_predict_flood_risk_from_postcodes():

    # Incomplete test - just checking it runs for now.

    DEFAULT_UNIT_DATA = tool.DEFAULT_UNIT_DATA
    idx = [1, 50, 673]

    postcodes = DEFAULT_UNIT_DATA.postcode.iloc[idx].tolist()
    predictions = test_tool.predict_flood_class_from_postcode(
        postcodes, model="all_minimum_risk"
    )

    # Should add assertions here.

    assert not predictions.empty


def test_predict_flood_risk_from_location():
    """Check flood risk prediction from location."""

    # Incomplete test - just checking it runs for now.

    eastings = [100000, 200000, 300000]
    northings = [500000, 600000, 700000]
    predictions = test_tool.predict_flood_class_from_OSGB36_location(
        eastings, northings, model="all_minimum_risk"
    )

    # Should add assertions here.

    assert not predictions.empty


def test_additional_functions():
    """Check other functions exist and run."""

    try:
        test_tool.predict_high_risk_near_watercourses(
            ["River Thames", "River Severn"],
            TEST_DATA.postcode.tolist(),
        )

    except NotImplementedError:
        pass

    try:
        test_tool.estimate_annual_human_flood_risk(
            TEST_DATA.postcode.tolist(),
        )

    except NotImplementedError:
        pass

    try:
        test_tool.estimate_annual_economic_flood_risk(
            TEST_DATA.postcode.tolist(),
        )

    except NotImplementedError:
        pass


# Convenience implementation to be able to run tests directly.
if __name__ == "__main__":
    test_lookup_easting_northing()
    test_lookup_lat_long()
