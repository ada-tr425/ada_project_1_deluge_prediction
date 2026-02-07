"""Analysis tools for flood data.

Functions to visualise and analyse flood risk and weather data."""

import math

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from ..utils import read_csv_from_example_data

__all__ = ["plot_postcode_density", "plot_risk_map"]

DEFAULT_POSTCODE_DATA = read_csv_from_example_data("postcodes_unlabelled.csv")


def plot_postcode_density(
    postcodes: str | pd.DataFrame = DEFAULT_POSTCODE_DATA,
    coordinate=["easting", "northing"],
    dx: float = 1000.0,
    **kwargs,
):
    """
    plot_postcode_density(postcodes=DEFAULT_POSTCODE_DATA,
                          coordinate=['easting', 'northing'],
                            dx=1000.0, **kwargs)

    Plot a postcode density map from a postcode file.


    Parameters
    ----------
    postcodese : str or pd.DataFrame, optional
        Path to postcode CSV file or DataFrame containing postcode data.
        Default is DEFAULT_POSTCODE_FILE.

    coordinate : list of str, optional
        List of two strings specifying the coordinate columns to use.
        Default is ['easting', 'northing'].

    dx : float, optional
        Grid spacing in coordinate units. Default is 1000.

    kwargs : dict, optional
        Additional keyword arguments to pass to matplotlib.pcolormesh.

    Returns
    -------

    matplotlib.axes.Axes
        The Axes object containing the plot.
    """

    if isinstance(postcodes, str):
        pdb = pd.read_csv(postcodes)
    else:
        pdb = postcodes.copy()

    bbox = (
        pdb[coordinate[0]].min() - 0.5 * dx,
        pdb[coordinate[0]].max() + 0.5 * dx,
        pdb[coordinate[1]].min() - 0.5 * dx,
        pdb[coordinate[1]].max() + 0.5 * dx,
    )

    nx = (
        math.ceil((bbox[1] - bbox[0]) / dx),
        math.ceil((bbox[3] - bbox[2]) / dx),
    )

    x = np.linspace(bbox[0] + 0.5 * dx, bbox[0] + (nx[0] - 0.5) * dx, nx[0])
    y = np.linspace(bbox[2] + 0.5 * dx, bbox[2] + (nx[1] - 0.5) * dx, nx[1])

    X, Y = np.meshgrid(x, y)

    Z = np.zeros(nx, int)

    for x, y in pdb[coordinate].values:
        Z[math.floor((x - bbox[0]) / dx), math.floor((y - bbox[2]) / dx)] += 1

    plt.pcolormesh(
        X,
        Y,
        np.where(Z > 0, Z, np.nan).T,
        norm=matplotlib.colors.LogNorm(),
        **kwargs,
    )
    plt.axis("equal")
    plt.colorbar()

    return plt.gca()


def plot_risk_map(risk_data, coordinate=["easting", "northing"], dx=10000):
    """
    Plot a risk map.

    Parameters
    ----------
    risk_data : pd.DataFrame
        DataFrame containing risk data with coordinate and riskLabel columns.
    coordinate : list of str, optional
        List of two strings specifying the coordinate columns to use.
        Default is ['easting', 'northing'].
    dx : float, optional
        Grid spacing in coordinate units. Default is 1000.
    Returns
    -------
    matplotlib.axes.Axes
        The Axes object containing the plot.
    """

    bbox = (
        risk_data[coordinate[0]].min() - 0.5 * dx,
        risk_data[coordinate[0]].max() + 0.5 * dx,
        risk_data[coordinate[1]].min() - 0.5 * dx,
        risk_data[coordinate[1]].max() + 0.5 * dx,
    )

    nx = (
        math.ceil((bbox[1] - bbox[0]) / dx),
        math.ceil((bbox[3] - bbox[2]) / dx),
    )

    x = np.linspace(bbox[0] + 0.5 * dx, bbox[0] + (nx[0] - 0.5) * dx, nx[0])
    y = np.linspace(bbox[2] + 0.5 * dx, bbox[2] + (nx[1] - 0.5) * dx, nx[1])

    X, Y = np.meshgrid(x, y)

    Z = np.zeros(nx, float)
    Zn = np.zeros(nx, int)

    for x, y, val in risk_data[coordinate + ["riskLabel"]].values:
        Z[math.floor((x - bbox[0]) / dx),
          math.floor((y - bbox[2]) / dx)] += val
        Zn[math.floor((x - bbox[0]) / dx),
           math.floor((y - bbox[2]) / dx)] += 1

    Z = Z / np.where(Zn > 0, Zn, 1)

    plt.pcolormesh(
        X, Y, np.where(Z > 0, Z, np.nan).T,
    )
    plt.axis("equal")
    plt.colorbar()

    return plt.gca()
