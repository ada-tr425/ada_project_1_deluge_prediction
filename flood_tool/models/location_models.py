import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder


class KNNLocalAuthorityModel:
    """
    A local authority prediction model based on KNN.
    This corresponds to the predict_local_authority interface.

    """
    def __init__(self, n_neighbors=1):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='ball_tree')
        self.le = LabelEncoder()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        X: A DataFrame containing 'easting' and 'northing'.
        y: Series containing the name of the local authority

        """
        # Remove NaN
        mask = ~y.isna()
        X_clean = X[mask]
        y_clean = y[mask]

        # labelencode
        y_encoded = self.le.fit_transform(y_clean)
        self.model.fit(X_clean, y_encoded)
        return self

    def predict(self, locations: np.ndarray) -> np.ndarray:
        """
        locations: An array of shape (n, 2) with columns [easting, northing]

        Returns: An array of predicted local authority names

        """
        pred_codes = self.model.predict(locations)
        return self.le.inverse_transform(pred_codes)


class KNNFloodRiskModel:
    """
    A KNN-based flood risk prediction model.

    Corresponding to the `predict_flood_class_from_OSGB36_location` interface.

    """
    def __init__(self, n_neighbors=10):
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance', algorithm='ball_tree')

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        X: A DataFrame containing 'easting' or 'northing'

        y: A Series containing risk levels (1-10 or 1-7)

        """
        mask = ~y.isna()
        self.model.fit(X[mask], y[mask])
        return self

    def predict(self, locations: np.ndarray) -> np.ndarray:
        """
        locations: An array of shape (n, 2) with columns [easting, northing]

        Returns: An array of predicted risk values

        """
        return self.model.predict(locations)