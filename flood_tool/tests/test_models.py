from flood_tool.models.baseline import (
    AllMinimumRiskModel,
    AllEnglandMedianPriceModel,
    ModalLocalAuthorityModel,
)

import pandas as pd


class TestAllMinimumRiskModel:

    def setup_method(self):
        # Setup code to initialize test environment
        self.model = AllMinimumRiskModel()

    def test_predict_from_postcodes(self):
        postcodes = ["AB1 2CD", "EF3 4GH", "IJ5 6KL"]
        predictions = self.model.predict_from_postcodes(postcodes)

        assert isinstance(predictions, pd.Series)
        assert predictions.dtype == int
        assert predictions.index.isin(postcodes).all()
        assert pd.Index(postcodes).isin(predictions.index).all()
        assert all(predictions == 1)
        assert predictions.max() <= 10
        assert predictions.min() >= 1

    def test_predict_from_locations(self):
        eastings = [100000, 200000, 300000]
        northings = [500000, 600000, 700000]
        locations = pd.MultiIndex.from_arrays(
            [eastings, northings], names=["easting", "northing"]
        )
        predictions = self.model.predict_from_locations(locations)

        assert isinstance(predictions, pd.Series)
        assert predictions.dtype == int
        assert predictions.index.equals(locations)
        assert all(predictions == 1)
        assert predictions.max() <= 10
        assert predictions.min() >= 1


class TestAllEnglandMedianPriceModel:

    def setup_method(self):
        # Setup code to initialize test environment
        self.model = AllEnglandMedianPriceModel()

    def test_predict_from_postcodes(self):
        postcodes = ["AB1 2CD", "EF3 4GH", "IJ5 6KL"]
        predictions = self.model.predict_from_postcodes(postcodes)

        assert isinstance(predictions, pd.Series)
        assert predictions.dtype == float
        assert predictions.index.isin(postcodes).all()
        assert pd.Index(postcodes).isin(predictions.index).all()
        assert all(predictions.dropna() >= 0)
        assert all(predictions.dropna() <= 1000_000_000)


class TestModalLocalAuthorityModel:

    def setup_method(self):
        # Setup code to initialize test environment
        self.model = ModalLocalAuthorityModel()

    def test_fit(self):

        (train_eastings,) = ([100000, 200000, 300000, 400000],)
        train_northings = [500000, 600000, 700000, 800000]
        local_authorities = [
            "AuthorityA",
            "AuthorityB",
            "AuthorityA",
            "AuthorityA",
        ]

        self.model.fit(
            pd.DataFrame(
                data={"easting": train_eastings, "northing": train_northings}
            ),
            pd.Series(local_authorities),
        )

    def test_predict_from_location(self):

        (train_eastings,) = ([100000, 200000, 300000, 400000],)
        train_northings = [500000, 600000, 700000, 800000]
        local_authorities = [
            "AuthorityA",
            "AuthorityB",
            "AuthorityA",
            "AuthorityA",
        ]

        self.model.fit(
            pd.DataFrame(
                data={"easting": train_eastings, "northing": train_northings}
            ),
            pd.Series(local_authorities),
        )

        test_eastings = [100000, 200000, 300000]
        test_northings = [500000, 600000, 700000]
        locations = pd.MultiIndex.from_arrays(
            [test_eastings, test_northings], names=["easting", "northing"]
        )

        predictions = self.model.predict_from_location(locations)

        assert isinstance(predictions, pd.Series)
        assert pd.api.types.is_object_dtype(predictions.dtype) or (
            pd.api.types.is_string_dtype(predictions.dtype)
        )
        assert predictions.index.isin(locations).all()
        assert pd.Index(locations).isin(predictions.index).all()
        assert all(predictions == "AuthorityA")
