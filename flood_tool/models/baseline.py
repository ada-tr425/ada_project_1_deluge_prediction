import numpy as np
import pandas as pd


# Could inherit from a common BaseModel if you want to enforce an interface
class AllFalseModel:

    def __init__(self, *args, **kwargs):
        """A baseline model that predicts minimum risk for all postcodes.

        No initialization or training is required."""

        del args, kwargs
        pass

    def fit(self, X, y, **kwargs):
        """No training required for this baseline model."""

        del X, y, kwargs

        return self

    def predict(self, X):

        return pd.Series(
            data=False,
            index=np.asarray(X),
            name="historicallyFlooded",
        )


class RandomBinaryClassifierModel:

    def __init__(self, seed=12345, *args, **kwargs):
        """A baseline model that predicts randomly True or False for
        all postcodes.

        No initialization or training is required."""

        self.generator = np.random.default_rng(seed)
        del args, kwargs
        pass

    def fit(self, X, y, **kwargs):
        """No training required for this baseline model."""

        del X, y, kwargs

        return self

    def predict(self, X):

        random_values = self.generator.integers(0, 2, size=len(X)).astype(bool)

        return pd.Series(
            data=random_values,
            index=np.asarray(X),
            name="historicallyFlooded",
        )


class AllMinimumRiskModel:

    def __init__(self, *args, **kwargs):
        """A baseline model that predicts minimum risk for all postcodes.

        No initialization or training is required."""

        del args, kwargs
        pass

    def fit(self, X, y, **kwargs):
        """No training required for this baseline model."""

        del X, y, kwargs

        return self

    def predict_from_postcodes(self, postcodes):
        return pd.Series(1, index=np.asarray(postcodes), name="riskLabel")

    def predict_from_locations(self, idx):

        return pd.Series(data=1, index=idx, name="riskLabel")


class AllEnglandMedianPriceModel:

    def __init__(self, *args, **kwargs):
        """A baseline model that predicts the median flood risk across England
        for all postcodes.

        No initialization or training is required."""

        del args, kwargs
        pass

    def fit(self, X, y, **kwargs):
        """No training required for this baseline model."""

        del X, y, kwargs

        return self

    def predict_from_postcodes(self, postcodes):
        return pd.Series(
            data=245000.00,
            index=np.asarray(postcodes),
            name="medianPrice",
        )


class ModalLocalAuthorityModel:

    def __init__(self, *args, **kwargs):
        """A baseline model that predicts the modal Local Authority
        for all postcodes.

        Model must be given postcode data to initialize."""

        del args, kwargs

    def fit(self, X, y, **kwargs):
        """Determine the modal Local Authority for the training data."""

        del X, kwargs

        self.modal_la = y.mode()[0]

        return self

    def predict_from_location(self, idx):
        return pd.Series(
            data=self.modal_la,
            index=idx,
            name="local_authority",
        )
