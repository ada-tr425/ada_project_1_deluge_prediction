import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    RobustScaler,
    FunctionTransformer,
    OneHotEncoder,
)
from sklearn.model_selection import train_test_split
from collections.abc import Sequence
from .utils import read_csv_from_preprocessed_data
from .models.house_prices_models import HousePricesXGBRegressor

# TODO: distance to london feature engineering from XY coordinates


def _drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Drop specified columns from the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    columns : list[str]
        A list of column names to drop.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the specified columns dropped.
    """
    df = df.copy()
    return df.drop(columns=columns, errors="ignore")


def _distance_to_london(easting: pd.Series, northing: pd.Series) -> pd.Series:
    """Calculate the distance to London from easting and northing coordinates.
    london_easting = 530000
    london_northing = 180000
    return np.sqrt((easting - london_easting) ** 2 +
                   (northing - london_northing) ** 2)

    NOTE: this function was tested, and it was found that there is no
    significant difference in model performance when including this feature.
    There was also high multicollinearity with easting and northing features.
    It is included here for completeness and potential future use.

    Parameters
    ----------
    easting : pd.Series
        A Series containing easting coordinates.
    northing : pd.Series
        A Series containing northing coordinates.

    Returns
    -------
    pd.Series
        A Series containing the distance to London for each coordinate pair.
    """
    london_easting = 530000
    london_northing = 180000
    return np.sqrt(
        (easting - london_easting) ** 2 + (northing - london_northing) ** 2
    )


def _split_num_cat_features(df: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    """
    Split the DataFrame columns into numerical and categorical features.

    Parameters
    ----------

    df : pandas.DataFrame
        The input DataFrame containing various features.

    Returns
    -------
    tuple[pd.Index, pd.Index]
        A tuple containing two pandas Index objects:
        - The first Index contains the names of numerical features.
        - The second Index contains the names of categorical features.
    """
    num_features = df.select_dtypes(include=np.number).columns
    cat_features = df.select_dtypes(exclude=np.number).columns
    return num_features, cat_features


def group_by_feature(district_data: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Group the district data by a specified column and aggregate features.

    Parameters
    ----------

    district_data : pandas.DataFrame
        The input DataFrame containing district-level features.
    column : str
        The column name to group by. Typically 'postcodeDistrict'
        or 'postcodeSector'.

    Returns
    -------
    pandas.DataFrame
        A DataFrame grouped by a specified column with numerical features
        aggregated by median and categorical features by mode.
    """
    df = district_data.copy()
    num_features, cat_features = _split_num_cat_features(df)

    df[num_features] = df.groupby(column)[num_features].transform("median")
    df[cat_features] = df.groupby(column)[cat_features].transform(
        lambda x: x.mode().iloc[0]
    )

    return df


def unit_district_merge(
    unit_data: pd.DataFrame, district_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepare input data by merging unit-level and district-level data.

    Parameters
    ----------
    unit_data : pandas.DataFrame
        The input DataFrame containing unit-level features.
    district_data : pandas.DataFrame
        The input DataFrame containing district-level features.

    Returns
    -------
    pandas.DataFrame
        A DataFrame resulting from merging unit-level and district-level data
        on 'postcodeDistrict'.
    """
    unit_data = unit_data.copy()
    district_data = district_data.copy()
    district_data = group_by_feature(district_data, "postcodeDistrict")

    merged_df = pd.merge(
        unit_data,
        district_data,
        on="postcodeDistrict",
        how="inner",
        suffixes=(None, "_district"),
    )

    return merged_df


def unit_sector_merge(
    unit_data: pd.DataFrame, sector_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepare input data by merging unit-level and sector-level data.

    Parameters
    ----------
    unit_data : pandas.DataFrame
        The input DataFrame containing unit-level features.
    sector_data : pandas.DataFrame
        The input DataFrame containing sector-level features.

    Returns
    -------
    pandas.DataFrame
        A DataFrame resulting from merging unit-level and sector-level data
        on 'postcodeSector'.
    """
    unit_data = unit_data.copy()
    sector_data = sector_data.copy()
    sector_data = group_by_feature(sector_data, "postcodeSector")

    merged_df = pd.merge(
        unit_data,
        sector_data,
        on="postcodeSector",
        how="inner",
        suffixes=(None, "_sector"),
    )

    return merged_df


def _inter_house_proc_pipeline(
    columns_to_drop: list[str], num_features: list[str]
) -> ColumnTransformer:
    """
    Create a processing pipeline for the merged postcode and
    sector-level data. Applies the following:

    1. Drops specified columns.
    2. Scales numerical features using RobustScaler.
    3. One-hot encodes categorical features.

    Parameters
    ----------
    columns_to_drop : list[str]
        A list of column names to drop from the DataFrame.
    num_features : list[str]
        A list of numerical feature names to scale.

    Returns
    -------
    ColumnTransformer
        A ColumnTransformer object that processes the merged postcode and
        sector-level data.
    """

    num_features = pd.Index(num_features)
    cat_features = pd.Index(["soilType"])

    column_dropper = FunctionTransformer(
        _drop_columns,
        kw_args={"columns": columns_to_drop},
        feature_names_out=None,
        validate=False,
    )

    num_pipe = Pipeline(steps=[("scaler", RobustScaler())])
    cat_pipe = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )

    proc_inter = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_features),
            ("cat", cat_pipe, cat_features),
        ],
        verbose_feature_names_out=False,
    )

    proc_pipe = Pipeline(
        steps=[("drop_columns", column_dropper), ("processing", proc_inter)]
    )

    return proc_pipe


def house_proc_pipeline(merge_with: str = "sector") -> pd.DataFrame:
    """
    Prepare a processing pipeline for working with postcode data merged
    with either district-level or sector-level data.

    Parameters
    ----------
    merge_with : str, optional
        The type of data to merge with: 'district' or 'sector'.
        Default is 'sector'.

    Returns
    -------
    proc_pipe : ColumnTransformer
        The fitted processing pipeline used for data transformation.
    """

    # Define columns to drop and numerical features for each merge type
    columns_to_drop = [
        "postcode",
        "postcodeDistrict_sector",
        "postcodeDistrict",
        "postcodeSector",
        "longitude",
        "latitude",
        "nearestWatercourse",
        "localAuthority",
        "riskLabel",
        "historicallyFlooded",
        "headcount",
    ]
    num_cols_dist = [
        "easting",
        "northing",
        "elevation",
        "distanceToWatercourse",
        "catsPerHousehold",
        "dogsPerHousehold",
        "pre_1900",
        "1900_1929",
        "1930_1945",
        "1945_1964",
        "1965_1982",
        "1983_1999",
        "2000_present",
        "unknown",
    ]
    num_cols_sect = [
        "easting",
        "northing",
        "elevation",
        "distanceToWatercourse",
        "households",
        "numberOfPostcodeUnits",
    ]

    # Merge and prepare processing pipeline based on merge_with parameter
    if merge_with == "district":
        proc_pipe = _inter_house_proc_pipeline(
            columns_to_drop=columns_to_drop, num_features=num_cols_dist
        )

    elif merge_with == "sector":
        proc_pipe = _inter_house_proc_pipeline(
            columns_to_drop=columns_to_drop, num_features=num_cols_sect
        )

    else:
        raise ValueError("merge_with must be either 'district' or 'sector'")

    proc_pipe.set_output(transform="pandas")

    return proc_pipe


def get_postcode_district_sector(postcodes: Sequence) -> pd.DataFrame:
    """
    get_postcode_district_sector(postcodes, dtype=np.float64)

    Get a Pandas dataframe containing formatted postcodes, postcode districts,
    and postcode sectors for sequence of postcodes of any format.

    Parameters
    ----------

    postcodes: sequence of strs
        Sequence of postcodes.

    Returns
    -------

    pandas.DataFrame
        DataFrame containing formatted postcodes, postcode sectors and
        postcode districts for the input postcodes.

    Examples
    --------
    >>> tool = Tool() # doctest: +SKIP
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

    DF = pd.DataFrame(
        {
            "postcode": Postcode_Formatted,
            "postcodeSector": Postcode_Sector,
            "postcodeDistrict": Postcode_District,
        }
    )

    return DF


def predict_house_price_from_postcodes_df(
    postcodes: Sequence[str],
) -> pd.DataFrame:
    """Predict house prices for a list of postcodes for plotting.

    Note this function is not an ideal implementation. It is intended as a
    workaround to allow plotting of house prices on maps in the visualization
    module.

    Parameters:
    ----------
    postcodes : list[str]
        A list of all postcodes for which predictions are required.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the predicted house prices for all
        postcodes, indexed by postcode with the input data associated with
        the postcodes.
    """

    # Preparing the data for training and fitting the model
    unit_prep = read_csv_from_preprocessed_data(
        "postcodes_impute_postcode_method.csv"
    )
    sector_prep = read_csv_from_preprocessed_data("sector_preprocessed.csv")

    merge_sector = unit_sector_merge(unit_prep, sector_prep)

    house_proc_pipe = house_proc_pipeline(merge_with="sector")

    X = merge_sector.drop(columns=["medianPrice"])
    y = merge_sector["medianPrice"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    X_train = house_proc_pipe.fit_transform(X_train)
    X_val = house_proc_pipe.transform(X_val)

    house_price_xgb = HousePricesXGBRegressor().fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    # Preparing the input data for prediction
    postcode_df = get_postcode_district_sector(postcodes)

    _unit = unit_prep.copy()
    _sector = sector_prep.copy()

    # Extract only relevant postcodes and sector data
    _unit = _unit[_unit["postcode"].isin(postcode_df["postcode"])]
    _sector = _sector[
        _sector["postcodeSector"].isin(postcode_df["postcodeSector"])
    ]

    # Merge unit and sector data
    merge_sector = unit_sector_merge(_unit, _sector)

    # Apply processing pipeline and drop target column if any
    X = house_proc_pipe.transform(merge_sector.drop(columns=["medianPrice"]))

    # Trim to only the postcodes found in the datasets
    postcodes_nan = postcode_df[
        ~postcode_df["postcode"].isin(merge_sector["postcode"])
    ]["postcode"].tolist()
    postcodes_pred = merge_sector["postcode"].tolist()

    y_pred = house_price_xgb.predict_from_postcodes(
        postcodes,
        postcode_pred=postcodes_pred,
        postcode_nan=postcodes_nan,
        X=X,
    )

    y_pred_df = merge_sector.copy()
    y_pred_df["medianPrice"] = y_pred[postcodes_pred].values

    return y_pred_df
