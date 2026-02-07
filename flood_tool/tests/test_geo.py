"""Test coordinate transformations."""

import flood_tool.geo as geo
import numpy as np

# unit tests for utility functions in geo module


class TestDMS2RAD:
    def test_dms2rad_degrees_only(self):
        """Test conversion with degrees only."""
        degrees = np.array([0, 90, 180])
        radians = np.array(
            [
                0,
                np.pi / 2,
                np.pi,
            ]
        )
        output = geo.dms2rad(degrees)
        assert np.allclose(radians, output)


def test_get_easting_northing_from_gps_lat_long_floats():
    """Test conversion from lat, long to osgb36 for floats."""
    lat = 55.5
    long = -1.54
    easting, northing = geo.get_easting_northing_from_gps_lat_long(lat, long)
    assert np.isclose(easting, 429157).all()
    assert np.isclose(northing, 623009).all()


def test_get_easting_northing_from_gps_lat_long_arrays():
    """Test conversion from lat, long to osgb36 for arrays."""
    lat = np.array([55.5])
    long = np.array([-1.54])
    easting, northing = geo.get_easting_northing_from_gps_lat_long(lat, long)
    assert np.isclose(easting, 429157).all()
    assert np.isclose(northing, 623009).all()


def test_get_gps_lat_long_from_easting_northing_floats():
    """Test conversion from easting, northing to WGS84 for floats."""
    easting = 422297.8
    northing = 412878.7
    lat, long = geo.get_gps_lat_long_from_easting_northing(easting, northing)
    assert np.isclose(lat, 53.612, rtol=1.0e-3).all()
    assert np.isclose(long, -1.664, rtol=1.0e-3).all()


def test_get_gps_lat_long_from_easting_northing_arrays():
    """Test conversion from easting, northing to WGS84 for floats."""
    easting = np.array([422297.8])
    northing = np.array([412878.7])
    lat, long = geo.get_gps_lat_long_from_easting_northing(easting, northing)
    assert np.isclose(lat, 53.612, rtol=1.0e-3).all()
    assert np.isclose(long, -1.664, rtol=1.0e-3).all()
