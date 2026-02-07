"""Example module in template package."""
from .models.location_models import KNNLocalAuthorityModel, KNNFloodRiskModel
from collections.abc import Sequence
from typing import List

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split


from .geo import *  # noqa: F401, F403
from .geo import get_gps_lat_long_from_easting_northing
from .models.seven_class_tool import SevenClassTool
from .utils import read_csv_from_resources, read_csv_from_example_data, \
                   read_csv_from_preprocessed_data
from .models import (
    AllMinimumRiskModel,
    AllEnglandMedianPriceModel,
    AllFalseModel,
    RandomBinaryClassifierModel,
    ModalLocalAuthorityModel,
    HousePricesXGBRegressor,
)
from .price_proc import (
    unit_district_merge,
    unit_sector_merge,
    house_proc_pipeline,
)

__all__ = [
    "Tool",
    "flood_class_from_postcode_models",
    "flood_class_from_location_models",
    "house_price_models",
    "local_authority_models",
    "historic_flooding_models",
]

# dictionaries with keys of short name and values of long name of
# classification/regression models

# You should add your own models in the relevant dictionary here
# in the format 'short_name': 'Long Name'
flood_class_from_postcode_models = {
    "all_minimum_risk": "All minimum risk",
    "seven_class_tool": "Seven Class Probability Model",
}
flood_class_from_location_models = {
    "all_minimum_risk": "All minimum risk",
    "seven_class_tool": "Seven Class Probability Model",
    "knn": "K-Nearest Neighbors",
}
historic_flooding_models = {
    "all_false": "All False",
    "random_classification": "Random Classification",
    "historic_rf": "Random Forest Historic Flood Classifier",
    
}
house_price_models = {    
    "house_prices_xgb": "HousePricesXGBRegressor",
    "all_england_median": "All England median price",
}
local_authority_models = {
    "modal_local_authority": "Modal Local Authority",
    "knn": "K-Nearest Neighbors", #add
}

IMPUTATION_CONSTANTS = {
    "soilType": "Unsurveyed/Urban",
    "elevation": 60.0,
    "nearestWatercourse": "Unknown",
    "distanceToWatercourse": 80,
    "localAuthority": "Unknown",
}

DEFAULT_FEATURE_DATA = read_csv_from_example_data("postcodes_unlabelled.csv")
DEFAULT_UNIT_DATA = read_csv_from_resources("postcodes_labelled.csv")
DEFAULT_SECTOR_DATA = read_csv_from_resources("sector_data.csv")
DEFAULT_DISTRICT_DATA = read_csv_from_resources("district_data.csv")
PREPROCESSED_UNIT_DATA = read_csv_from_preprocessed_data("postcodes_impute_postcode_method.csv")
PREPROCESSED_UNIT_DATA_KNN = read_csv_from_preprocessed_data("postcodes_impute_KNN.csv")
PREPROCESSED_SECTOR_DATA = read_csv_from_preprocessed_data("sector_preprocessed.csv")
PREPROCESSED_DISTRICT_DATA = read_csv_from_preprocessed_data("district_imputed.csv")
PREPROCESSED_RAINFALL_STATION = read_csv_from_preprocessed_data("rainfall_stations.csv")
PREPROCESSED_LEVEL_STATION = read_csv_from_preprocessed_data("level_stations_imputed.csv")
PREPROCESSED_WET_DAY = read_csv_from_example_data("wet_day_preprocessed.csv")
PREPROCESSED_TYPICAL_DAY = read_csv_from_example_data("typical_day_preprocessed.csv")

RAW_FEATURES = [
    "postcode",
    "easting",
    "northing",
    "soilType",
    "elevation",
    "nearestWatercourse",
    "distanceToWatercourse",
    "localAuthority",
]


class Tool(object):
    """
    Tool(X=DEFAULT_FEATURE_DATA, unit_data=DEFAULT_UNIT_DATA,
                sector_data=DEFAULT_SECTOR_DATA,
                district_data=DEFAULT_DISTRICT_DATA,
                unit_prep=PREPROCESSED_UNIT_DATA,
                unit_prep_KNN=PREPROCESSED_UNIT_DATA_KNN,
                sector_prep=PREPROCESSED_SECTOR_DATA,
                district_prep=PREPROCESSED_DISTRICT_DATA,
                rainfall_stations=PREPROCESSED_RAINFALL_STATION
                level_stations=PREPROCESSED_LEVEL_STATION
                additional_data={})

    Class to interact with a postcode database file.
    """

    def __init__(
        self,
        X: pd.DataFrame | None = DEFAULT_FEATURE_DATA,
        unit_data: pd.DataFrame = DEFAULT_UNIT_DATA,
        sector_data: pd.DataFrame = DEFAULT_SECTOR_DATA,
        district_data: pd.DataFrame = DEFAULT_DISTRICT_DATA,
        unit_prep: pd.DataFrame = PREPROCESSED_UNIT_DATA,
        unit_prep_knn: pd.DataFrame = PREPROCESSED_UNIT_DATA_KNN,
        sector_prep: pd.DataFrame = PREPROCESSED_SECTOR_DATA,
        district_prep: pd.DataFrame = PREPROCESSED_DISTRICT_DATA,
        rainfall_stations: pd.DataFrame = PREPROCESSED_RAINFALL_STATION,
        level_stations: pd.DataFrame = PREPROCESSED_LEVEL_STATION,
        additional_data: dict = {},
    ):
        """
        Parameters
        ----------
        X: DataFrame, optional
            DataFrame containing unlabelled feature data (see the example
            unlabelled postcode data file for the expected format) for use
            in models and lookups.

        unit_data: DataFrame, optional
            DataFrame containing features and class labels for specific
            postcodes.

        sector_data : DataFrame, optional
            DataFrame containing information on households by postcode sector.

        district_data : DataFrame, optional
            DataFrame containing information on households by postcode
            district.

        unit_prep: DataFrame, optional
            DataFrame containing features and class labels for specific
            postcodes, preprocessed with imputation using postcodes method.

        unit_prep_knn: DataFrame, optional
            DataFrame containing features and class labels for specific
            postcodes, preprocessed with imputation using KNN method.

        sector_prep : DataFrame, optional
            DataFrame containing information on households by postcode sector, preprocessed.

        district_prep : DataFrame, optional
            DataFrame containing information on households by postcode district, preprocessed.

        rainfall_stations : DataFrame, optional
            DataFrame containing information on rainfall stations, preprocessed.

        level_stations : DataFrame, optional
            DataFrame containing information on water level stations, preprocessed.

        additional_data: dict, optional
            Dictionary containing additional DataFrames with additional
            information on households.
        """

        self.unit_data = unit_data
        self.sector_data = sector_data
        self.district_data = district_data
        self.additional_data = additional_data
        self.unit_prep = unit_prep
        self.unit_prep_knn = unit_prep_knn
        self.sector_prep = sector_prep
        self.district_prep = district_prep
        self.rainfall_stations = rainfall_stations
        self.level_stations = level_stations

        self._postcodedb = unit_data[RAW_FEATURES].copy()

        self.X = X

        if X is not None:
            self._postcodedb = pd.concat([X, self._postcodedb], axis=0)
            self._postcodedb = self._postcodedb.drop_duplicates(
                subset=["postcode"], keep="first"
            )

        self.models = {}

        # continue your work here

    def fit_to_data(
        self,
        unit_data: pd.DataFrame | None = None,
        models: str | List[str] = [],
        update_hyperparameters: bool = False,
        **kwargs,
    ):
        """
        fit_to_data(unit_data=None, models=[], update_hyperparameters=False
                    **kwargs)

        Fit/train model or models using the initialised data, optionally
        updating training set and hyperparameters.

        Parameters
        ----------
        unit_data : DataFrame, optional
            DataFrame containing additional features and class labels for
            specific postcodes.
        models : str or list of model keys
            Models to fit/train
        update_hyperparameters : bool, optional
            If True, models may tune their hyperparameters, where
            possible. If False, models will use their default hyperparameters.
        **kwargs : dict, optional
            Additional keyword arguments to pass to model fitting methods.
        """

        # Perform any necessary updates to the data

        if unit_data is not None:
            self.unit_data = pd.concat([self.unit_data, unit_data], axis=0)
            self.unit_data = self.unit_data.drop_duplicates(
                subset=["postcode"], keep="last"
            )

            self._postcodedb = pd.concat(
                [self._postcodedb, unit_data[RAW_FEATURES]], axis=0
            )

            self._postcodedb = self._postcodedb.drop_duplicates(
                subset=["postcode"], keep="last"
            )

        if isinstance(models, str):
            models = [models]

        X_loc = self.unit_data[["easting", "northing"]]

        # Next fit/train each model in turn
        for model in models:
            if update_hyperparameters:
                print(f"tuning {model} hyperparameters")
                # Do your hyperparameter tuning for the specified model
            else:
                print(f"training {model}")
                # Do your regular fitting/training for the specified model

                if model == "random_classification":
                    # Example of fitting a model
                    X = self.unit_data[["easting", "northing"]]
                    y = self.unit_data["historicallyFlooded"]
                    self.models[model] = RandomBinaryClassifierModel(
                        seed=42
                    ).fit(X, y)

                elif model == "modal_local_authority":
                    # Example of fitting a model
                    X = self._postcodedb[["easting", "northing"]]
                    y = self._postcodedb["localAuthority"]
                    self.models[model] = ModalLocalAuthorityModel().fit(X, y)

                elif model == "seven_class_tool":
                    model_instance = SevenClassTool(label_col="riskLabel").fit(self.unit_data)
                    model_instance.set_reference_data(self._postcodedb)
                    self.models["seven_class_tool"] = model_instance



                elif model == "house_prices_xgb":
                    merge_sector = unit_sector_merge(self.unit_prep,
                                                     self.sector_prep)
    
                    self.house_proc_pipe = house_proc_pipeline(
                        merge_with='sector')

                    X = merge_sector.drop(columns=['medianPrice'])
                    y = merge_sector['medianPrice']

                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=0.1, random_state=42)
                    
                    X_train = self.house_proc_pipe.fit_transform(X_train)
                    X_val = self.house_proc_pipe.transform(X_val)

                    self.models[model] = HousePricesXGBRegressor().fit(
                        X_train, y_train,
                        eval_set=[(X_train, y_train), (X_val, y_val)],
                        verbose=False
                        )
                
                elif model == "historic_rf":

                    from .models.historic_flood_classifier import build_historic_rf

                    df_train = self.unit_prep.copy()  
                    self.models[model] = build_historic_rf(df_train)


                elif model == "knn":
                    self.models["knn_authority"] = KNNLocalAuthorityModel(n_neighbors=5).fit(
                        X_loc, self.unit_data["localAuthority"]
                    )
                    self.models["knn_risk"] = KNNFloodRiskModel(n_neighbors=15).fit(
                        X_loc, self.unit_data["riskLabel"]
                    )

    def lookup_easting_northing(
        self, postcodes: Sequence, dtype: np.dtype = np.float64
    ) -> pd.DataFrame:
        """
        lookup_easting_northing(postcodes, dtype=np.float64)

        Get a dataframe of OS eastings and northings from a sequence of
        input postcodes in the labelled or unlabelled datasets.
        """

        postcodes = pd.Index(postcodes)

        frame = self._postcodedb.copy()
        frame = frame.set_index("postcode")
        frame = frame.reindex(postcodes)

        return frame.loc[postcodes, ["easting", "northing"]].astype(dtype)

    def lookup_lat_long(
        self, postcodes: Sequence, dtype: np.dtype = np.float64
    ) -> pd.DataFrame:
        """
        lookup_lat_long(postcodes, dtype=np.float64)

        Get a Pandas dataframe containing GPS latitude and longitude
        information for a sequence of postcodes in the labelled or
        unlabelled datasets.
        """

        DF_east_north = self.lookup_easting_northing(postcodes)
        lats, lons = get_gps_lat_long_from_easting_northing(
                     DF_east_north['easting'], DF_east_north['northing'])
        DF = pd.DataFrame({"longitude": lons, "latitude": lats}, \
                          index=postcodes, dtype=dtype)
        return DF

    def get_postcode_district_sector(self, postcodes: Sequence) -> pd.DataFrame:
        """
        get_postcode_district_sector(postcodes)

        Get a Pandas dataframe containing formatted postcodes, postcode districts, 
        and postcode sectors for sequence of postcodes of any format.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing formatted postcodes, postcode sectors and postcode districts 
            for the input postcodes.

        Examples
        --------
        >>> tool = Tool()
        >>> tool.get_postcode_district_sector['M34 7QL']) # doctest: +SKIP
                postcode  postcodeSector  postcodeDistrict
        postcode
        M34 7QL  M34 7QL           M34 7               M34
        """

        Postcode_District, Postcode_Sector, Postcode_Formatted = [], [], []

        for i in range(len(postcodes)):
            Postcode = postcodes[i]
            if " " in Postcode:
                Split = Postcode.split()
                Outward = Split[0]
                Inward = Split[1]
            else:
                Inward = Postcode[-3:]
                Outward = Postcode[:-3].strip("_").strip("-").strip("+")
            Postcode_District.append(Outward.upper())
            Postcode_Sector.append(str(Outward + " " + Inward[0]).upper())
            Postcode_Formatted.append(str(Outward + " " + Inward).upper())

        DF = pd.DataFrame({"postcode": Postcode_Formatted, "postcodeSector": Postcode_Sector,
                          "postcodeDistrict": Postcode_District})

        return DF

    def get_station_coordinates(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        get_station_coordinates(dataframe)

        Get a Pandas dataframe containing longitude, latitude, easting and northing coordinates
        of rainfall and water level stations based on stationReference.

        Parameters
        ----------

        dataframe : pandas.DataFrame
            DataFrame (in the format of typical_day.csv or wet_day.csv)

        Returns
        -------

        pandas.DataFrame
            A copy of the input DataFrame with new columns of
            longitude, latitude, easting and northing coordinates.
            Original DataFrame remains unmodified.

        Examples
        --------
        >>> tool = Tool()
        >>> data = read_csv_from_example_data('wet_day.csv')
        >>> data_coords = tool.get_station_coordinates(data)  # doctest: +SKIP
        """

        DF = dataframe.copy()
        lons = np.zeros(len(DF))
        lats = np.zeros(len(DF))
        easts = np.zeros(len(DF))
        norths = np.zeros(len(DF))
        values = np.zeros(len(DF))

        # Find station using stationReference
        for i in range(len(DF)):
            station_ref = DF['stationReference'][i]
            if DF['parameter'][i] == 'rainfall':
                station_row = self.rainfall_stations[self.rainfall_stations['stationReference'] == station_ref].reset_index()
            elif DF['parameter'][i] == 'level':
                station_row = self.level_stations[self.level_stations['stationReference'] == station_ref].reset_index()
            # Get coordinates if station can be found in database
            if len(station_row) > 0:
                lons[i] = station_row['longitude'][0]
                lats[i] = station_row['latitude'][0]
                easts[i] = station_row['easting'][0]
                norths[i] = station_row['northing'][0]
            # np.nan if station cannot be found in database
            else:
                lons[i], lats[i], easts[i], norths[i] = np.nan, np.nan, np.nan, np.nan

            # Clean values column
            try:
                values[i] = float(DF['value'][i])
            except:
                Value = DF['value'][i].split("|")
                values[i] = Value[0]

        # Update columns
        DF['latitude'] = lats
        DF['longitude'] = lons
        DF['easting'] = easts
        DF['northing'] = norths

        # Drop rows with no geographic coordinates found
        DF = DF.dropna(subset=['latitude', 'longitude']).reset_index().drop(columns='index')

        return DF

    def handle_outliers_iqr(self, df: pd.DataFrame, cols: str | list[str]) -> pd.DataFrame:
        """
        Handle outliers in one or more DataFrame columns using the Interquartile Range (IQR) method.
        Outliers are capped at lower and upper bounds derived from the 1st quartile (Q1), 
        3rd quartile (Q3), and IQR (1.5*IQR rule).

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing the column(s) to process.
        cols : str or list of str
            Single column name (str) or list of column names to handle outliers for.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with outliers in specified column(s) capped.
            Original DataFrame remains unmodified.

        Examples
        --------
        >>> tool = Tool()
        >>> import pandas as pd
        >>> df = pd.DataFrame({'a': [1, 2, 3, 100, -50], 'b': [5, 6, 7, 200, -100]})
        >>> # Handle single column
        >>> cleaned_single = tool.handle_outliers_iqr(df, 'a')
        >>> # Handle multiple columns
        >>> cleaned_multi = tool.handle_outliers_iqr(df, ['a', 'b'])  # doctest: +SKIP
        """
        # Create a copy to avoid modifying the original DataFrame
        df_cleaned = df.copy()

        # Standardize input to list format for uniform processing
        cols_list = [cols] if isinstance(cols, str) else cols

        for col in cols_list:
            # Calculate quartiles and IQR
            q1 = df_cleaned[col].quantile(0.25)
            q3 = df_cleaned[col].quantile(0.75)
            iqr = q3 - q1

            # Calculate outlier bounds
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Cap outliers at the calculated bounds
            df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)

        return df_cleaned

    def impute_missing_values(
        self,
        dataframe: pd.DataFrame,
        method: str = "postcode",
        constant_values: dict = IMPUTATION_CONSTANTS,
        k: int = 5,
    ) -> pd.DataFrame:
        """
        impute_missing_values(dataframe, method='postcode',
                                constant_values=IMPUTATION_CONSTANTS)

        Impute missing values in an unlabelled dataframe.

        Parameters
        ----------

        dataframe : pandas.DataFrame
            DataFrame (in the format of the unlabelled postcode data)
            potentially containing missing values as NaNs, or with missing
            columns.

        method : str, optional
            Method to use for imputation. Options include:
            * ``'postcode'``, impute using median (numerical features) or mode (categorical features)
            of other datapoints within the same postcode sector or postcode district in the labelled dataset.
            * ``'knn'``, impute using median (numerical features) or mode (categorical features)
            of the k-nearest neighbours based on geographic coordinates from the labelled dataset.
            * ``'constant'``, impute using a constant value for imputation.

        constant_values : dict, optional
            Dictionary containing constant values to use for imputation in the format 
            ``{column_name: value}``.
            Only used if method is ``'constant'``.

        Returns
        -------

        pandas.DataFrame
            DataFrame with missing values imputed.

        Examples
        --------

        >>> tool = Tool()
        >>> data = read_csv_from_example_data('postcodes_missing_data.csv')
        >>> full_data = tool.impute_missing_values(data)  # doctest: +SKIP
        """

        DF = dataframe.copy()
        unit_data = self.unit_data.copy()

        # Find columns with missing values that need imputing
        missing = DF.isnull().sum() / len(DF)
        columns_impute = list(missing[missing > 0.0].index)
        # Separate columns into numerical and categorical features
        columns_impute_num = list(DF[columns_impute].select_dtypes(include=[np.number]).columns)
        columns_impute_cat = list(DF[columns_impute].select_dtypes(exclude=[np.number]).columns)

        # Postcode method
        if method == 'postcode':
            # First get postcode sectors and postcode districts
            unit_data['postcodeSector'] = self.get_postcode_district_sector(unit_data['postcode'])['postcodeSector']
            unit_data['postcodeDistrict'] = self.get_postcode_district_sector(unit_data['postcode'])['postcodeDistrict']
            DF['postcodeSector'] = self.get_postcode_district_sector(DF['postcode'])['postcodeSector']
            DF['postcodeDistrict'] = self.get_postcode_district_sector(DF['postcode'])['postcodeDistrict']
            #
            # Numerical columns
            if len(columns_impute_num) > 0:
                # Get medians of each postcode sector and postcode district from labelled dataset
                sector_medians = unit_data.groupby("postcodeSector")[columns_impute_num].median()
                district_medians = unit_data.groupby("postcodeDistrict")[columns_impute_num].median()
                # Merge medians into new dataset
                DF = DF.merge(sector_medians, on="postcodeSector", how="left", suffixes=("", "_sector"))
                DF = DF.merge(district_medians, on="postcodeDistrict", how="left", suffixes=("", "_district"))
                for col in columns_impute_num:
                    # First try imputing with medians of postcode sector
                    DF[col] = DF[col].fillna(DF[col+"_sector"])
                    # Then try imputing with medians of postcode district
                    DF[col] = DF[col].fillna(DF[col+"_district"])
                    # If all else fails, impute with median of entire dataset
                    DF[col] = DF[col].fillna(unit_data[col].median())
                    # Remove merging columns
                    DF = DF.drop(columns=[col+"_sector", col+"_district"])
            #
            # Categorical columns
            if len(columns_impute_cat) > 0:
                # Get modes of each postcode sector and postcode district from labelled dataset
                sector_modes = unit_data.dropna(subset=columns_impute_cat)\
                            .groupby("postcodeSector")[columns_impute_cat].agg(lambda x: x.mode().iloc[0])
                district_modes = unit_data.dropna(subset=columns_impute_cat)\
                            .groupby("postcodeDistrict")[columns_impute_cat].agg(lambda x: x.mode().iloc[0])
                # Merge modes into new dataset
                DF = DF.merge(sector_modes, on="postcodeSector", how="left", suffixes=("", "_sector"))
                DF = DF.merge(district_modes, on="postcodeDistrict", how="left", suffixes=("", "_district"))
                for col in columns_impute_cat:
                    # First try imputing with modes of postcode sector
                    DF[col] = DF[col].fillna(DF[col+"_sector"])
                    # Then try imputing with modes of postcode district
                    DF[col] = DF[col].fillna(DF[col+"_district"])
                    # If all else fails, use default imputation constants
                    DF[col] = DF[col].fillna(constant_values[col])
                    # Remove merging columns
                    DF = DF.drop(columns=[col+"_sector", col+"_district"])

        # knn method
        elif method == 'knn':
            # Impute based on easting northing coordinates
            coords_labelled = np.array(unit_data[['easting', 'northing']])
            coords_new = np.array(DF[['easting', 'northing']])
            # Fit nearest neighbours
            neighbors = NearestNeighbors(n_neighbors=k).fit(coords_labelled)
            #
            # Numerical columns
            if len(columns_impute_num) > 0:
                for col in columns_impute_num:
                    vals = DF[col].values.copy()
                    for i in range(len(DF)):
                        if pd.isna(vals[i]):
                            # Find nearest neighbours within the labelled dataset
                            dists, indexes = neighbors.kneighbors([coords_new[i]])
                            neighbor_vals = pd.Series(unit_data[col][indexes[0]]).dropna()
                            # Impute with median of k neighbours
                            if len(neighbor_vals) > 0:
                                vals[i] = np.nanmedian(neighbor_vals)
                            # If no data for all k neighbours, impute with median of entire dataset
                            else:
                                vals[i] = np.nanmedian(unit_data[col])
                    # Replace column with imputed values
                    DF[col] = vals
            #
            # Categorical columns
            if len(columns_impute_cat) > 0:
                for col in columns_impute_cat:
                    vals = DF[col].values.copy()
                    for i in range(len(DF)):
                        if pd.isna(vals[i]):
                            # Find nearest neighbours within the labelled dataset
                            dists, indexes = neighbors.kneighbors([coords_new[i]])
                            neighbor_vals = pd.Series(unit_data[col][indexes[0]]).dropna()
                            # Impute with mode of k neighbours
                            if len(neighbor_vals) > 0:
                                vals[i] = neighbor_vals.mode().iloc[0]
                            # If no data for all k neighbours, use default imputation constants
                            else:
                                vals[i] = constant_values[col]
                    # Replace column with imputed values
                    DF[col] = vals

        # Constant method
        elif method == 'constant':
            for col in columns_impute:
                Impute_Constant = constant_values[col]
                DF[col] = DF[col].fillna(Impute_Constant)

        return (DF)

    def impute_district_property_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing district-level property age values for rows with all null targets.
        Trains a multi-output regression model using predefined features, 
        then predicts/fills null targets.

        Predefined columns:
            - Features: ['catsPerHousehold', 'dogsPerHousehold']
            - Targets: ['pre_1900', '1900_1929', '1930_1945', '1945_1964', 
                       '1965_1982', '1983_1999', '2000_present']

        Parameters
        ----------
        df : pandas.DataFrame
            District DataFrame with required feature/target columns (outliers handled).

        Returns
        -------
        pandas.DataFrame
            District DataFrame with imputed values for rows missing all targets.

        Raises
        ------
        ValueError
            If input lacks required feature/target columns.

        Examples
        --------
        >>> tool = Tool()
        >>> df_imputed = tool.impute_district_property_age(df)   # doctest: +SKIP
        """
        # Predefined columns for district property age imputation
        features = ['catsPerHousehold', 'dogsPerHousehold']
        targets = ['pre_1900', '1900_1929', '1930_1945', '1945_1964', 
                   '1965_1982', '1983_1999', '2000_present', 'unknown']

        # Check for required columns
        missing_features = [f for f in features if f not in df.columns]
        missing_targets = [t for t in targets if t not in df.columns]
        if missing_features or missing_targets:
            raise ValueError(
                f"Missing required columns. Features: {missing_features}; Targets: {missing_targets}"
            )

        # Split data: train (non-null targets), test (all null targets)
        df_train = df.dropna(subset=targets, how='all')
        df_test = df[df[targets].isna().all(axis=1)]

        if df_test.empty:
            return df.copy()

        # Train model and predict
        model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
        model.fit(df_train[features], df_train[targets])
        predictions = model.predict(df_test[features])

        # Normalize and fill predictions
        df_test_pred = pd.DataFrame(predictions, columns=targets, index=df_test.index)
        df_test_pred = df_test_pred.div(df_test_pred.sum(axis=1), axis=0)  # Row-wise normalization

        df_imputed = df.copy()
        df_imputed.loc[df_test.index, targets] = df_test_pred

        return df_imputed

    def predict_flood_class_from_postcode(
        self, postcodes: Sequence[str], model: str = "all_minimum_risk"
    ) -> pd.Series:
        """
        Generate series predicting flood probability classification
        for a collection of poscodes.
        """

        if model == "all_minimum_risk":
            model_instance = AllMinimumRiskModel()  # No training required
            return model_instance.predict_from_postcodes(postcodes)


        elif model == "seven_class_tool":
            if "seven_class_tool" not in self.models:
                raise RuntimeError(
                    "Model 'seven_class_tool' has not been trained. "
                )

            model_instance = self.models["seven_class_tool"]

            postcodes_idx = pd.Index(postcodes, name="postcode")
            frame = self._postcodedb.set_index("postcode")
            X_raw = frame.reindex(postcodes_idx)

            return model_instance.predict_from_features(
                X_raw, index=postcodes_idx
            )
        else:
            raise NotImplementedError(f"model {model} not implemented")

    def predict_flood_class_from_OSGB36_location(
        self,
        eastings: Sequence[float],
        northings: Sequence[float],
        model: str = "all_minimum_risk",
    ) -> pd.Series:
        """
        Generate series predicting flood probability classification
        for a collection of locations given as eastings and northings
        on the Ordnance Survey National Grid (OSGB36) datum.
        """
        idx = pd.MultiIndex.from_arrays(
                [eastings, northings], names=["easting", "northing"])

        if model == "all_minimum_risk":
            idx = pd.MultiIndex.from_arrays(
                [eastings, northings], names=["easting", "northing"]
            )
            model_instance = AllMinimumRiskModel()  # No training required
            return model_instance.predict_from_locations(idx)

        elif model == "seven_class_tool":
            if "seven_class_tool" not in self.models:
                raise RuntimeError(
                    "Model 'seven_class_tool' has not been trained."
                )

            model_instance = self.models["seven_class_tool"]

            # Build raw DataFrame
            df = pd.DataFrame({
                "easting": eastings,
                "northing": northings,
            })

            # Convert OSGB36 to WGS84
            lat, lon = get_gps_lat_long_from_easting_northing(
                df["easting"].values,
                df["northing"].values,
                dms=False
            )

            df["latitude"] = lat
            df["longitude"] = lon

            preds = model_instance.predict_from_features(df, index=idx)
            return preds

        elif model == "knn":
            if "knn_risk" not in self.models:
                raise ValueError("Model 'knn' not trained. Call fit_to_data(['knn']) first.")
            
            X_pred = np.column_stack([eastings, northings])
            
            predictions_float = self.models["knn_risk"].predict(X_pred)
            
            predictions = np.round(predictions_float).astype(int)
            

            return pd.Series(data=predictions, index=idx, name="riskLabel")
            
        else:
            raise NotImplementedError(f"model {model} not implemented")

    def predict_flood_class_from_WGS84_locations(
        self,
        longitudes: Sequence[float],
        latitudes: Sequence[float],
        model: str = "all_minimum_risk",
    ) -> pd.Series:
        """
        Generate series predicting flood probability classification
        for a collection of WGS84 datum locations.
        """

        if model == "all_minimum_risk":
            idx = pd.MultiIndex.from_arrays(
                [longitudes, latitudes], names=["longitude", "latitude"]
            )
            model_instance = AllMinimumRiskModel()  # No training required
            return model_instance.predict_from_locations(idx)

        elif model == "seven_class_tool":
            if "seven_class_tool" not in self.models:
                raise RuntimeError(
                    "Model 'seven_class_tool' has not been trained."
                )

            model_instance = self.models["seven_class_tool"]

            # Build raw DataFrame
            df = pd.DataFrame({
                "longitude": longitudes,
                "latitude": latitudes,
            })

            preds = model_instance.predict_from_features(df, index=idx)
            return preds

        else:
            raise NotImplementedError(f"model {model} not implemented")

    def predict_median_house_price(
        self, postcodes: Sequence[str], model: str = "all_england_median"
    ) -> pd.Series:
        """
        Generate a Pandas Series containing predicted median house price for an
        input sequence of postcodes.
        """

        postcode_df = self.get_postcode_district_sector(postcodes)

        if model == "all_england_median":
            model_instance = AllEnglandMedianPriceModel()
            return model_instance.predict_from_postcodes(postcodes)


        elif model == "house_prices_xgb":
            # Prepare the data needed for prediction
            _unit = self.unit_prep.copy()
            _sector = self.sector_prep.copy()

            # Extract only relevant postcodes and sector data
            _unit = _unit[_unit['postcode'].isin(
                postcode_df['postcode'])]
            _sector = _sector[_sector['postcodeSector'].isin(
                postcode_df['postcodeSector'])]

            # Merge unit and sector data
            merge_sector = unit_sector_merge(_unit, _sector)

            # Apply processing pipeline and drop target column if any
            if merge_sector.shape[0] == 0:
                X = merge_sector.drop(columns=['medianPrice'])
            else:
                X = self.house_proc_pipe.transform(merge_sector.drop(
                    columns=['medianPrice']))

            # Trim to only the postcodes found in the datasets
            postcodes_nan = postcode_df[~postcode_df['postcode'].isin(
                merge_sector['postcode'])]['postcode'].tolist()
            postcodes_pred = merge_sector['postcode'].tolist()

            return self.models[model].predict_from_postcodes(
                postcodes, postcodes_pred, postcodes_nan, X)

        else:
            raise NotImplementedError(f"model {model} not implemented")

    def predict_high_risk_near_watercourses(
        self,
        watercourse_names: Sequence[str],
        postcodes: Sequence[str] | None = None,
        risk_above: int = 4,
    ) -> pd.DataFrame:
        """
        Generate a DataFrame of unit areas with flood risk above a threshold
        that are near specified watercourses.
        """

        # Load the dataset
        df = self.X
        
        # Select the aimed postcode
        if postcodes is not None:
            df = df[df['postcode'].isin(postcodes)]
        if df.empty:
            return pd.DataFrame(columns=['postcode', 'riskLabel'] + list(df.columns))
        
        # Call the risk prediction function to obtain the flood risk level for each location
        easting = df['easting'].tolist()
        northing = df['northing'].tolist()
        risk_series = self.predict_flood_class_from_OSGB36_location(easting, northing)

        # Merge risk labels into data sets
        risk_df = risk_series.reset_index().rename(columns={0: 'riskLabel'})
        df = df.merge(risk_df, on=['easting', 'northing'], how='inner')

        # Screen for records that are high risk and close to designated waterways
        df = df[df['riskLabel'] > risk_above]
        df = df[df['nearestWatercourse'].isin(watercourse_names)]

        # Organize result columns
        result_columns = [col for col in df.columns if col != 'riskLabel'] + ['riskLabel']
        df = df[result_columns]

        return df

    def predict_local_authority(
        self,
        eastings: Sequence[float],
        northings: Sequence[float],
        model: str = "modal_local_authority",
    ) -> pd.Series:
        """
        Generate a Pandas Series predicting local authorities for a sequence
        of OSGB36 locations.
        """
        idx = pd.MultiIndex.from_arrays(
            [eastings, northings], names=["easting", "northing"])

        if model == "modal_local_authority":
            idx = pd.MultiIndex.from_tuples(
                [(est, nth) for est, nth in zip(eastings, northings)]
            )
            return self.models["modal_local_authority"].predict_from_location(
                idx
            )
        
        elif model == "knn":
            if "knn_authority" not in self.models:
                raise ValueError("Model 'knn' not trained. Call fit_to_data(['knn']) first.")
            
            X_pred = np.column_stack([eastings, northings])
            predictions = self.models["knn_authority"].predict(X_pred)
            
            return pd.Series(data=predictions, index=idx, name="localAuthority")
            
        else:
            raise NotImplementedError(f"model {model} not implemented")

    def predict_historic_flooding(
        self, postcodes: Sequence[str], model: str = "all_false"
    ) -> pd.Series:
        """
        Generate series predicting whether a collection of postcodes
        has experienced historic flooding.
        """

        if model == "all_false":
            model_instance = AllFalseModel()  # No training required
            return model_instance.predict(postcodes)
        elif model == "random_classification":
            return self.models["random_classification"].predict(postcodes)
        
        
        elif model == "historic_rf":
            
            if "historic_rf" not in self.models:
                raise RuntimeError(
                    "Model 'historic_rf' has not been trained. "
                    "Call tool.fit_to_data(models='historic_rf') first."
                )
            
            from .models.historic_flood_classifier import FEATURE_COLS, DECISION_THRESHOLD

            
            postcodes_idx = pd.Index(postcodes, name="postcode")
            frame = self.unit_prep.set_index("postcode")

            X = frame.reindex(postcodes_idx)[FEATURE_COLS]

            model_instance = self.models["historic_rf"]
            y_proba = model_instance.predict_proba(X)[:, 1]
            y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)

            return pd.Series(
                y_pred,
                index=postcodes_idx,
                name="historicallyFlooded_pred",
            )


        else:
            raise NotImplementedError(f"model {model} not implemented")
        




    def estimate_total_value(self, postal_data: Sequence[str]) -> pd.Series:
        """
        Return a series of estimates of the total property values
        of a sequence of postcode units or postcode sectors.
        """
        # Load the datasets
        postcodes_labelled = self.unit_data
        sector_data = self.sector_data
    
        # Build mapping relationship
        valid_prices = postcodes_labelled[
            postcodes_labelled['medianPrice'].notna()
        ]
        price_map = valid_prices.set_index('postcode')['medianPrice'].to_dict()

        valid_sectors = sector_data[
            sector_data['households'].notna()
        ]
        sector_households_map = valid_sectors.set_index('postcodeSector')['households'].to_dict()

        total_values = []
        for postcode in postal_data:
            # Get the median house price of the current zip code
            median_price = price_map.get(postcode, 0.0)
            
            # Extract zip code partitions and match the number of households
            sector = self.get_postcode_district_sector([postcode]).iloc[0]['postcodeSector']
            households = sector_households_map.get(sector, 0)
            
            # Calculate total value
            total_value = median_price * households
            total_values.append(total_value)
        
        # Build result Series
        return pd.Series(
            data=total_values,
            index=pd.Index(postal_data, name='postcode'),
            name='total_property_value'
        )


    def estimate_annual_human_flood_risk(
        self, postcodes: Sequence[str], risk_labels: pd.Series | None = None
    ) -> pd.Series:
        """
        Return a series of estimates of the risk to human life for a
        collection of postcodes.
        """

        risk_labels = risk_labels or self.predict_flood_class_from_postcode(
            postcodes
        )

        # Make sure to align with the input postcode index
        risk_labels = risk_labels.reindex(postcodes)

        # Load population data
        sector_data = self.sector_data
        
        # Build mapping relationship
        sector_headcount = sector_data.set_index("postcodeSector")["headcount"].to_dict()

        # Define flood levels
        flood_probability_mapping = {
            7: 0.05,   # 5%+
            6: 0.03,   # 3%
            5: 0.02,   # 2%
            4: 0.01,   # 1%
            3: 0.005,  # 0.5%
            2: 0.002,  # 0.2%
            1: 0.001   # 0.1% or less
        }

        annual_risks = []
        for postcode in postcodes:
            # Extract zip code partition (such as BA10 0AD → BA10 0)
            sector = self.get_postcode_district_sector([postcode]).iloc[0]['postcodeSector']
            
            # Get the total population
            headcount = sector_headcount.get(sector, 0)
            
            # Get the flood probability corresponding to the risk level
            risk_level = risk_labels.get(postcode, 0)
            flood_prob = flood_probability_mapping.get(risk_level, 0)
            
            # Calculate annual risk
            annual_risk = 0.1 * headcount * flood_prob
            annual_risks.append(annual_risk)
        
        return pd.Series(
            data=annual_risks,
            index=pd.Index(postcodes, name="postcode"),
            name="annual_human_flood_risk"
        )


    def estimate_annual_economic_flood_risk(
        self, postcodes: Sequence[str], risk_labels: pd.Series | None = None
    ) -> pd.Series:
        """
        Return a series of estimates of the total economic property risk
        for a collection of postcodes.
        """

        risk_labels = risk_labels or self.predict_flood_class_from_postcode(
            postcodes
        )

        # Make sure to align with the input postcode index
        risk_labels = risk_labels.reindex(postcodes)
        
        # Get total property value
        total_value_series = self.estimate_total_value(postcodes)

        # Define flood levels
        flood_probability_mapping = {
            7: 0.05,   # 5%+
            6: 0.03,   # 3%
            5: 0.02,   # 2%
            4: 0.01,   # 1%
            3: 0.005,  # 0.5%
            2: 0.002,  # 0.2%
            1: 0.001   # 0.1% or less
        }

        annual_economic_risks = []
        for postcode in postcodes:
            # Get total property value
            total_value = total_value_series.get(postcode, 0.0)
            
            # Get the flood probability corresponding to the risk level
            risk_level = risk_labels.get(postcode, 0)
            flood_prob = flood_probability_mapping.get(risk_level, 0)
            
            # Calculate financial risk
            economic_risk = 0.05 * total_value * flood_prob
            annual_economic_risks.append(economic_risk)

        return pd.Series(
            data=annual_economic_risks,
            index=pd.Index(postcodes, name="postcode"),
            name="annual_economic_flood_risk"
        )
