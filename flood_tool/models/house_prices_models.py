import numpy as np
import pandas as pd
import xgboost as xgb
from collections.abc import Sequence

# TODO: optimize hyperparameters


class HousePricesXGBRegressor(xgb.XGBRegressor):
    """An XGBoost regression model to predict house prices at utilizing
    postcode and one of (sector-level or district-level) data.

    Inherits from xgboost's XGBRegressor.
    """

    def __init__(self):
        """Initialize the XGBoost regression model."""

        super().__init__(
            early_stopping_rounds=15,
            eval_metric='rmse'
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> pd.DataFrame:
        """Fit the XGBoost regression model to the merged dataset.

        Parameters
        ----------
        X : pandas.DataFrame
            The input features for training containing merged postcode,
            and one of (sector-level or district-level) data.
        y : pandas.Series
            The target variable representing house prices with log1p
            transformation applied.

        Returns
        -------
        XGBoost model
            The fitted model.
        """
        # Apply log1p transformation to the target variable
        y = np.log1p(y)

        # Handle eval_set if provided, and apply log1p transformation to the
        # target variable in each eval set
        eval_set = kwargs.pop('eval_set', None)
        if eval_set is not None:
            eval_set_transformed = []
            for X_eval, y_eval in eval_set:
                y_eval_transformed = np.log1p(y_eval)
                eval_set_transformed.append((X_eval, y_eval_transformed))
            kwargs['eval_set'] = eval_set_transformed

        return super().fit(X, y, **kwargs)

    def predict(self, X: pd.DataFrame, postcodes=None) -> pd.Series:
        """Predict house prices for the given input features.

        Parameters
        ----------
        X : pandas.DataFrame
            The input features for prediction containing merged postcode,
            and sector-level data.
        postcodes : array-like, optional
            The postcodes corresponding to the input features.

        Returns
        -------
        pd.Series
            A Series containing the predicted house prices, indexed by
            postcodes if provided. Applies an expm1 transformation to
            revert the log1p transformation.
        """
        y_pred = super().predict(X)
        y_pred = np.expm1(y_pred)

        # Check if postcodes are provided for indexing
        if postcodes is not None:
            return pd.Series(
                y_pred,
                index=np.asarray(postcodes),
                name="medianPrice")
        else:
            return pd.Series(y_pred, name="medianPrice")

    def predict_from_postcodes(self, postcodes: Sequence[str],
                               postcode_pred: Sequence[str],
                               postcode_nan: Sequence[str],
                               X: pd.DataFrame) -> pd.Series:
        """Predict house prices for a list of postcodes, handling missing data.

        Parameters:
        ----------
        postcodes : list[str]
            A list of all postcodes for which predictions are required.
        postcode_pred : list[str]
            A list of postcodes that have sector/district data available
            for prediction.
        postcode_nan : list[str]
            A list of postcodes that do not have sector/district data
            available.
        X : pd.DataFrame
            The input features for prediction containing merged postcode,
            and sector/district-level data.

        Returns:
        -------
        pd.Series
            A Series containing the predicted house prices for all
            postcodes, with fallback values for those without data.
        """
        # To be used for postcodes without sector/district data
        # median_house_price = np.full(len(postcode_nan), np.median(y_pred))
        median_house_price = np.full(len(postcode_nan), 245000.00)
        y_nan = pd.Series(
            median_house_price,
            index=np.asarray(postcode_nan),
            name="medianPrice"
        )
        if X.shape[0] == 0:
            # If there are no postcodes with data, return only the fallback
            return y_nan.reindex(postcodes)

        # Predict for postcodes with sector/district data
        y_pred = self.predict(X, postcodes=postcode_pred)

        # Combine predictions and fallback values into one Series
        y = pd.concat([y_pred, y_nan]).reindex(postcodes)

        return y
