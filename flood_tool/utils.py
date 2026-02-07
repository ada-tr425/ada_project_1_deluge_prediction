"""File containing utility functions for the flood_tool package."""

import os
import pandas as pd

__all__ = [
    "read_csv_from_resources",
    "read_csv_from_example_data",
    "read_csv_from_preprocessed_data",
    "_data_dir",
    "_example_data_dir",
    "_preprocessed_data_dir",
]

_data_dir = os.path.join(os.path.dirname(__file__), "resources")
_example_data_dir = os.path.join(_data_dir, "example_data")
_preprocessed_data_dir = os.path.join(_data_dir, "preprocessed_data")


def read_csv_from_resources(filename: str, **kwargs) -> pd.DataFrame:
    filepath = os.path.join(_data_dir, filename)
    return pd.read_csv(filepath, **kwargs)


def read_csv_from_example_data(filename: str, **kwargs) -> pd.DataFrame:
    filepath = os.path.join(_example_data_dir, filename)
    return pd.read_csv(filepath, **kwargs)


def read_csv_from_preprocessed_data(filename: str, **kwargs) -> pd.DataFrame:
    """Read a CSV file from the preprocessed_data directory."""
    filepath = os.path.join(_preprocessed_data_dir, filename)
    return pd.read_csv(filepath, **kwargs)
