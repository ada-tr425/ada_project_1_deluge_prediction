"""
Mapping utilities for visualising flood models.

This module is only about *visualisation*:

- It does NOT train any models.
- It expects you to pass in data (for example, a DataFrame with
  latitude/longitude and model predictions).
- Functions here are usually called from `analysis.py` or from
  Jupyter notebooks.
"""

from __future__ import annotations
import itertools
import numpy as np
import folium
import pandas as pd
import geopandas as gpd
from pathlib import Path
import branca.colormap as cm
from folium.plugins import Fullscreen, MiniMap, MeasureControl, MousePosition
from ..models.historic_flood_classifier import predict_postcode

# ---------------------------------------------------------------------------
# General colour palette for Folium markers
FOLIUM_COLORS = [
    "red",
    "blue",
    "green",
    "purple",
    "orange",
    "darkred",
    "lightred",
    "beige",
    "darkblue",
    "darkgreen",
    "cadetblue",
    "darkpurple",
    "white",
    "pink",
    "lightblue",
    "lightgreen",
    "gray",
    "black",
    "lightgray",
]
# ---------------------------------------------------------------------------
__all__ = [
    "make_base_map",
    "make_interactive_base_map",
    "_make_colour_lookup",
    "plot_historic_flood_for_postcode",
    "plot_flood_by_area",
    "plot_local_authority_map",
    "plot_local_authority_real_shapes",
    "plot_color_size_map",
    "filtered_plot",
    "plot_latest_rainfall_and_level",
    "latest_weather_data",
    "plot_combined_flood_and_rain_map",
]

# ---------------------------------------------------------------------------
# General mapping helpers


def make_base_map(
    centre_lat: float,
    centre_lon: float,
    zoom_start: int = 12,
) -> folium.Map:
    """
    Create a simple Folium map centred on a single point.

    Parameters
    ----------
    centre_lat : float
        Centre latitude.
    centre_lon : float
        Centre longitude.
    zoom_start : int, default 12
        Initial zoom level.

    Returns
    -------
    folium.Map
        A basic Folium map with a scale bar.
    """
    return folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=zoom_start,
        control_scale=True,
    )


# -------------------------------------------------------------


def make_interactive_base_map(
    df: pd.DataFrame,
    *,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    zoom_start: int = 8,
    width: str = "55%",
    height: str = "380px",
) -> folium.Map:
    """
    Create an interactive base map centred on the mean location of a DataFrame.

    The map includes:
    - Fullscreen toggle
    - Mini overview map
    - Distance measurement tool
    - Live mouse coordinates

    Parameters
    ----------
    df : pd.DataFrame
        Data with latitude and longitude columns.
    lat_col, lon_col : str, default "latitude", "longitude"
        Column names for coordinates.
    zoom_start : int, default 8
        Initial zoom level.
    width, height : str
        Map size as strings understood by Folium (e.g. "55%", "380px").

    Returns
    -------
    folium.Map
        Interactive base map ready to add layers.
    """
    m = folium.Map(
        location=[df[lat_col].mean(), df[lon_col].mean()],
        zoom_start=zoom_start,
        width=width,
        height=height,
        control_scale=True,
    )

    Fullscreen(position="topleft").add_to(m)
    MiniMap(toggle_display=True, position="bottomleft").add_to(m)
    MeasureControl(position="topleft", primary_length_unit="meters").add_to(m)
    MousePosition(
        position="bottomleft",
        separator=" | ",
        num_digits=5,
        prefix="Lat/Lon:",
    ).add_to(m)

    return m


# ---------------------------------------------------------------------------
def _make_colour_lookup(labels, colours=FOLIUM_COLORS):
    """
    Assign a colour to each unique label.

    Parameters
    ----------
    labels : sequence
        Labels to colour (e.g. local authorities, postcode areas).
    colours : list of str
        List of colour names to cycle through.

    Returns
    -------
    dict
        Mapping {label: colour}.
    """
    labels = list(pd.unique(labels))
    colour_cycle = itertools.cycle(colours)
    return {lab: next(colour_cycle) for lab in labels}


def latest_weather_data(
    df: pd.DataFrame,
    parameter: str = "rainfall",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> folium.Map:
    """
    Plot the latest rainfall or river level for each station.

    The function:
    - Filters the data for a chosen parameter (e.g. "rainfall" or "level")
    - Keeps only the most recent reading for each station
    - Colours markers by value using simple quartile-based bins

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with measurement values and coordinates.
        Must contain:
        - 'dateTime'
        - 'value'
        - 'parameter'
        - 'stationReference'
        plus latitude/longitude columns.
    parameter : str, default "rainfall"
        Parameter to plot (e.g. "rainfall", "level").
    lat_col, lon_col : str
        Column names for latitude and longitude.

    Returns
    -------
    folium.Map
        A Folium map with one circle marker per station.
    """
    df = df.copy()

    # Ensure datetime and value are correctly typed
    df["dateTime"] = pd.to_datetime(df["dateTime"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Filter by chosen parameter
    df_param = df[
        df["parameter"]
        .astype(str)
        .str.contains(parameter, case=False, na=False)
    ]
    if df_param.empty:
        raise ValueError(f"No rows found for parameter '{parameter}'")

    # Pick latest measurement for each station
    df_latest = df_param.sort_values("dateTime").drop_duplicates(
        subset=["stationReference"], keep="last"
    )
    df_latest = df_latest.dropna(subset=[lat_col, lon_col, "value"])

    if df_latest.empty:
        raise ValueError(
            "No valid rows after filtering/cleaning "
            f"for parameter '{parameter}'."
        )

    # Simple value-based colour bins using quartiles
    q25 = df_latest["value"].quantile(0.25)
    q50 = df_latest["value"].quantile(0.50)
    q75 = df_latest["value"].quantile(0.75)

    def colour_for_value(v: float) -> str:
        if v >= q75:
            return "red"  # highest values
        elif v >= q50:
            return "orange"  # high
        elif v >= q25:
            return "green"  # moderate
        else:
            return "blue"  # low

    # Create map (roughly centred over England)
    m = folium.Map(location=[51.0, 0.0], zoom_start=7)

    # Add markers
    for _, row in df_latest.iterrows():
        value = float(row["value"])
        colour = colour_for_value(value)
        popup_html = (
            f"<b>Station:</b> {row['stationReference']}<br>"
            f"<b>{parameter.title()} Value:</b> {value}<br>"
            f"<b>Time:</b> {row['dateTime']}"
        )
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=6,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.8,
            popup=popup_html,
        ).add_to(m)

    # Mini legend (static)
    legend_html = f"""
    <div style="
        position:absolute; bottom:20px; left:20px; z-index:9999;
        background:rgba(255,255,255,0.9); padding:8px 10px;
        border-radius:6px; border:1px solid #999; font-size:12px;">
        <b>Latest {parameter.title()} value</b><br>
        <span style="color:blue;">●</span> low<br>
        <span style="color:green;">●</span> moderate<br>
        <span style="color:orange;">●</span> high<br>
        <span style="color:red;">●</span> very high<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


# ---------------------------------------------------------------------------
# Model 1: Historic flood classifier – postcode area (BA, BN, BH, .)


def plot_historic_flood_for_postcode(
    postcode: str,
    *,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> folium.Map:
    """
    Run the historic flood classifier for a single postcode and plot it.

    This uses `predict_postcode(postcode)` from
    `flood_tool.models.historic_flood_classifier` and then shows:

    - The true historic flood label
    - The predicted label
    - The predicted probability

    Parameters
    ----------
    postcode : str
        Full postcode string (e.g. "BH12 1AA").
    lat_col, lon_col : str
        Column names for latitude and longitude in the returned DataFrame.

    Returns
    -------
    folium.Map
        A small map centred on the postcode with one coloured marker.
    """
    df = predict_postcode(postcode)

    if df.empty:
        raise ValueError(f"No data found for postcode '{postcode}'.")

    row = df.iloc[0]

    if lat_col not in df.columns or lon_col not in df.columns:
        raise KeyError(
            f"Columns '{lat_col}' and/or '{lon_col}' not found in dataframe. "
            "Check the column names in the preprocessed dataset "
            "'postcodes_impute_KNN.csv'."
        )

    lat = float(row[lat_col])
    lon = float(row[lon_col])

    fmap = make_base_map(lat, lon, zoom_start=13)

    pred = int(row["predictedHistoricFlooded"])
    colour = "red" if pred == 1 else "blue"

    true_label = int(row["historicallyFlooded"])
    prob = float(row["predictedFloodProb"])

    popup_html = (
        f"<b>Postcode:</b> {row['postcode']}<br>"
        f"<b>True label (historicallyFlooded):</b> {true_label}<br>"
        f"<b>Predicted:</b> {pred}<br>"
        f"<b>Predicted probability (class 1):</b> {prob:.3f}"
    )

    folium.CircleMarker(
        location=[lat, lon],
        radius=8,
        color=colour,
        fill=True,
        fill_color=colour,
        fill_opacity=0.8,
        popup=folium.Popup(popup_html, max_width=300),
    ).add_to(fmap)

    return fmap


# ---------------------------------------------------------------------------
# Model 1: Historic flood classifier – postcode areas (BA, BN, BH, …)
# ---------------------------------------------------------------------------


def plot_flood_by_area(
    df: pd.DataFrame,
    *,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    district_col: str = "postcodeDistrict",
    flood_col: str = "historicallyFlooded",
    areas: list[str] | None = None,
    width: str = "55%",
    height: str = "380px",
) -> folium.Map:
    """
    Plot historic flooding for all postcode units, grouped by *postcode area*.

    The postcode area is the leading alphabetic part only, e.g.:
    - "BA" for "BA1 1AA"
    - "BN" for "BN10 7AA"

    Each area becomes its own layer (FeatureGroup) that you can toggle on/off.

    Parameters
    ----------
    df : pd.DataFrame
        Data with postcode, flood labels and coordinates.
    lat_col, lon_col : str
        Column names for latitude and longitude.
    district_col : str
        Column containing the postcode district (e.g. "BA1", "BN10").
    flood_col : str
        Column containing the historic flood label.
    areas : list of str, optional
        If given, only show these postcode areas (e.g. ["BA", "BN"]).
    width, height : str
        Map size passed to Folium.

    Returns
    -------
    folium.Map
        Interactive map with a layer per postcode area.
    """
    df = df.copy()

    # Extract postcode area (leading letters) from postcodeDistrict
    df["postcodeArea"] = df[district_col].astype(str).str.extract(r"^([A-Z]+)")

    if areas is not None:
        df = df[df["postcodeArea"].isin(areas)]

    if df.empty:
        raise ValueError("No data available for the selected postcode areas.")

    m = folium.Map(
        location=[df[lat_col].mean(), df[lon_col].mean()],
        zoom_start=8,
        width=width,
        height=height,
        control_scale=True,
    )

    Fullscreen(position="topleft").add_to(m)
    MiniMap(toggle_display=True, position="bottomleft").add_to(m)
    MeasureControl(position="topleft", primary_length_unit="meters").add_to(m)
    MousePosition(
        position="bottomleft",
        separator=" | ",
        num_digits=5,
        prefix="Lat/Lon:",
    ).add_to(m)

    area_list = sorted(df["postcodeArea"].dropna().unique())
    counts = df["postcodeArea"].value_counts().to_dict()

    # Assign colours to areas
    if len(area_list) <= len(FOLIUM_COLORS):
        area_colors = {
            area: FOLIUM_COLORS[i] for i, area in enumerate(area_list)
        }
    else:
        # Fall back to HSL colours if there are many areas
        def hsl(i, n):
            return f"hsl({int(360 * i / max(1, n))}, 70%, 45%)"

        area_colors = {
            area: hsl(i, len(area_list)) for i, area in enumerate(area_list)
        }

    # Add one FeatureGroup per area
    for area in area_list:
        fg = folium.FeatureGroup(name=f"{area} ({counts.get(area, 0)})")
        colour = area_colors[area]
        df_area = df[df["postcodeArea"] == area]

        for r in df_area.itertuples(index=False):
            historically_flooded = getattr(r, flood_col, "N/A")
            district = getattr(r, district_col, "")
            sector = getattr(r, "postcodeSector", "")

            popup_html = f"""
                <b>{r.postcode}</b><br>
                Area: {area}<br>
                District: {district}<br>
                Sector: {sector}<br>
                HistoricallyFlooded: {historically_flooded}
            """

            folium.CircleMarker(
                location=[getattr(r, lat_col), getattr(r, lon_col)],
                radius=5,
                color=colour,
                fill=True,
                fill_opacity=0.75,
                tooltip=f"{r.postcode} – {area}",
                popup=popup_html,
            ).add_to(fg)

        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Zoom to data extent
    m.fit_bounds(
        [
            [df[lat_col].min(), df[lon_col].min()],
            [df[lat_col].max(), df[lon_col].max()],
        ]
    )

    # Simple legend
    legend_rows = ""
    for area in area_list:
        color = area_colors[area]
        cnt = counts.get(area, 0)
        legend_rows += f"""
            <div style="display:flex;align-items:center;margin:3px 0;">
                <span style="
                    width:12px;height:12px;flex:0 0 12px;
                    background:{color};
                    border:1px solid #555;border-radius:2px;
                    display:inline-block;margin-right:6px;
                "></span>
                <span style="font-size:12px;line-height:1;">
                    {area} <span style="color:#666;">({cnt})</span>
                </span>
            </div>
        """

    legend_html = f"""
    <div id="legend-areas" style="
        position:absolute;
        bottom:16px;
        right:16px;
        z-index:9999;
        background:rgba(255,255,255,0.95);
        padding:8px 10px;
        border-radius:6px;
        border:1px solid rgba(0,0,0,0.25);
        box-shadow:0 2px 8px rgba(0,0,0,0.15);
        font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,
        sans-serif;
        ">
        <div style="display:flex;justify-content:space-between;
                    align-items:center;gap:12px;margin-bottom:6px;
                    cursor:pointer;"
             onclick="var c=this.nextElementSibling;
                      c.style.display=(c.style.display==='none'?'block':'none');">
            <div style="font-weight:600;font-size:13px;">Postcode Areas</div>
            <div style="font-size:11px;color:#555;">toggle</div>
        </div>
        <div style="max-height:150px;overflow:auto;padding-right:2px;">
            {legend_rows}
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


# ---------------------------------------------------------------------------
# Model 2: Local Authority Model and Flood risk Visualization
# ---------------------------------------------------------------------------


def plot_local_authority_map(
    df: pd.DataFrame,
    *,
    la_col: str,
    easting_col: str = "easting",
    northing_col: str = "northing",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    width: str = "55%",
    height: str = "380px",
    zoom_start: int = 8,
    fmap: folium.Map | None = None,
) -> folium.Map:
    """
    Plot local authority information as coloured points.

    This version works directly from a DataFrame (e.g. from a CSV) rather
    than using shapefiles or GeoPackages.

    Parameters
    ----------
    df : pd.DataFrame
        Data with local authority labels and coordinates.
    la_col : str
        Column name containing the local authority label.
    easting_col, northing_col : str
        Kept for reference; not used directly in the map.
    lat_col, lon_col : str
        Column names for latitude and longitude.
    width, height : str
        Map size if a new map is created.
    zoom_start : int
        Zoom level for a new map.
    fmap : folium.Map, optional
        Existing map to add this layer to. If None, a new map is created.

    Returns
    -------
    folium.Map
        Map with a "Local authority" layer added.
    """
    if la_col not in df.columns:
        raise KeyError(f"Column '{la_col}' not found in DataFrame.")

    df_plot = df.copy()

    # Use existing map, or create a new one
    if fmap is None:
        m = make_interactive_base_map(
            df_plot,
            lat_col=lat_col,
            lon_col=lon_col,
            zoom_start=zoom_start,
            width=width,
            height=height,
        )
    else:
        m = fmap

    colour_lookup = _make_colour_lookup(df_plot[la_col])

    # Separate layer for local authority points
    fg = folium.FeatureGroup(name="Local authority", show=True)

    for r in df_plot.itertuples(index=False):
        la = getattr(r, la_col)
        colour = colour_lookup[la]
        lat = getattr(r, lat_col)
        lon = getattr(r, lon_col)
        postcode = getattr(r, "postcode", "")

        popup_html = (
            f"<b>Local authority:</b> {la}<br>"
            f"<b>Postcode:</b> {postcode}<br>"
            f"Lat: {lat:.5f}, Lon: {lon:.5f}"
        )

        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=colour,
            fill=True,
            fill_opacity=0.75,
            popup=popup_html,
            tooltip=f"{postcode} – {la}" if postcode else la,
        ).add_to(fg)

    fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ---------------------------------------------------------------------------
# Model 2: Local Authority + flood risk – polygon and point visualisation
# ---------------------------------------------------------------------------


def plot_local_authority_real_shapes(
    df_units: pd.DataFrame,
    *,
    la_col: str = "localAuthority",
    boundary_path: str | Path = None,
    boundary_name_col: str = "LAD23NM",
    risk_col: str = "riskLabel",
    postcode_col: str = "postcode",
    hist_col: str = "historicallyFlooded",
    zoom_start: int = 7,
) -> folium.Map:
    """
    Plot local authorities using official boundaries and overlay flooded
    points.

    - Local authority polygons come from a GeoPackage (ONS LAD 2023).
    - Only postcodes with historicallyFlooded == 1 are plotted as points.
    - Points are coloured by flood risk.

    Parameters
    ----------
    df_units : pd.DataFrame
        Model output at postcode level, including:
        - local authority name
        - latitude / longitude
        - risk label
        - historic flood indicator
    la_col : str, default "localAuthority"
        Column name with local authority names in `df_units`.
    boundary_path : str or Path
        Path to the boundary GeoPackage (.gpkg) file.
    boundary_name_col : str, default "LAD23NM"
        Column in the boundary file with local authority names.
    risk_col : str, default "riskLabel"
        Column with flood risk category/score.
    postcode_col : str, default "postcode"
        Column with postcode strings, used in popups.
    hist_col : str, default "historicallyFlooded"
        Column with 0/1 indicator for historic flooding.
    zoom_start : int, default 7
        Initial zoom level.

    Returns
    -------
    folium.Map
        Map with:
        - a polygon layer for local authorities
        - a point layer for flooded postcodes
    """
    # 1. Validate boundary file path
    if boundary_path is None:
        raise ValueError("You must provide boundary_path to the .gpkg file.")

    boundary_path = Path(boundary_path)
    if not boundary_path.exists():
        raise FileNotFoundError(f"Boundary file not found at: {boundary_path}")

    # 2. Load boundaries
    gdf_bound = gpd.read_file(boundary_path)
    if gdf_bound.crs is not None and gdf_bound.crs.to_string() != "EPSG:4326":
        gdf_bound = gdf_bound.to_crs("EPSG:4326")

    # 2b. Work with a copy of units & keep only historically flooded == 1
    df = df_units.copy()

    if hist_col not in df.columns:
        raise KeyError(f"Column '{hist_col}' not found in df_units.")

    # Keep only rows where historicallyFlooded == 1
    df = df[df[hist_col] == 1].copy()

    if df.empty:
        raise ValueError(f"No rows with '{hist_col} == 1' – nothing to plot.")

    # Clean names for matching
    df[la_col] = df[la_col].astype(str).str.strip()
    gdf_bound[boundary_name_col] = (
        gdf_bound[boundary_name_col].astype(str).str.strip()
    )

    # 3. Filter to authorities present in df (only those with histFlood=1)
    las_predicted = df[la_col].unique()
    gdf_subset = gdf_bound[
        gdf_bound[boundary_name_col].isin(las_predicted)
    ].copy()

    if len(gdf_subset) == 0:
        raise ValueError(
            "No matching local authority names between model and boundary "
            "file."
        )

    # 4. Colour polygons per authority
    colors = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "darkred",
        "lightred",
        "beige",
        "darkblue",
        "darkgreen",
        "cadetblue",
        "darkpurple",
        "white",
        "pink",
        "lightblue",
        "lightgreen",
        "gray",
        "black",
        "lightgray",
    ]

    la_list = sorted(gdf_subset[boundary_name_col].unique())
    color_map = {la: colors[i % len(colors)] for i, la in enumerate(la_list)}

    def style_function(feature):
        la_name = feature["properties"][boundary_name_col]
        col = color_map.get(la_name, "lightgray")
        return {
            "fillColor": col,
            "color": col,
            "weight": 1,
            "fillOpacity": 0.45,
        }

    # 5. Base map
    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)

    # Local authority polygons
    folium.GeoJson(
        gdf_subset,
        name="Local Authorities",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=[boundary_name_col],
            aliases=["Local Authority:"],
        ),
    ).add_to(m)

    # 6. Flood-risk points layer (only histFlood == 1 rows)
    def risk_color(r):
        """
        Map a numeric risk score to a colour.

        Assumes risk values like 1–6 where higher means higher risk.
        """
        r = int(r)
        mapping = {
            0: "lightgray",
            1: "green",
            2: "yellow",
            3: "orange",
            4: "red",
            5: "darkred",
            6: "purple",
        }
        return mapping.get(r, "black")

    points_layer = folium.FeatureGroup(name="Flood risk points", show=True)

    for _, row in df.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        risk = row[risk_col]

        popup_parts = []
        if postcode_col in df.columns:
            popup_parts.append(f"Postcode: {row[postcode_col]}")
        popup_parts.append(f"Risk label: {risk}")
        popup_parts.append(
            f"Historically flooded: {row[hist_col]}"
        )  # always 1 now

        popup_html = "<br>".join(popup_parts)
        col = risk_color(risk)

        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color=col,
            fill=True,
            fill_color=col,
            fill_opacity=0.7,
            weight=0.3,
            popup=popup_html,
        ).add_to(points_layer)

    points_layer.add_to(m)
    folium.LayerControl().add_to(m)
    return m


# ----------------------------------------------------------------------------------------
# Rainfall and Water Level Overlay Layer
# ----------------------------------------------------------------------------------------


def add_latest_weather_layer(
    fmap: folium.Map,
    df: pd.DataFrame,
    *,
    parameter: str = "rainfall",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    show: bool = False,
    layer_name: str | None = None,
) -> folium.Map:
    """
    Add the latest rainfall or river level readings as a new layer on a map.

    This is the same idea as `plot_latest_weather_data`, but instead of
    creating a new map, it adds a FeatureGroup to an existing one.

    Parameters
    ----------
    fmap : folium.Map
        Existing map (e.g. from `plot_flood_by_area`).
    df : pd.DataFrame
        Dataset with:
        - 'dateTime'
        - 'value'
        - 'parameter'
        - 'stationReference'
        plus latitude/longitude columns.
    parameter : str, default "rainfall"
        Parameter to plot (e.g. "rainfall", "level").
    lat_col, lon_col : str
        Column names for latitude and longitude.
    show : bool, default False
        Whether this layer is visible by default.
    layer_name : str, optional
        Name in the layer control. Defaults to "Latest <parameter>".

    Returns
    -------
    folium.Map
        The same map, with an extra weather layer added.
    """
    df = df.copy()

    # Clean types
    df["dateTime"] = pd.to_datetime(df["dateTime"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Filter by parameter
    df_param = df[
        df["parameter"]
        .astype(str)
        .str.contains(parameter, case=False, na=False)
    ]
    if df_param.empty:
        raise ValueError(f"No rows found for parameter '{parameter}'")

    # Latest per station
    df_latest = df_param.sort_values("dateTime").drop_duplicates(
        subset=["stationReference"], keep="last"
    )
    df_latest = df_latest.dropna(subset=[lat_col, lon_col, "value"])

    if df_latest.empty:
        raise ValueError(
            "No valid rows after filtering/cleaning "
            f"for parameter '{parameter}'."
        )

    # Value-based bins using quartiles (same idea as plot_latest_weather_data)
    q25 = df_latest["value"].quantile(0.25)
    q50 = df_latest["value"].quantile(0.50)
    q75 = df_latest["value"].quantile(0.75)

    def colour_for_value(v: float) -> str:
        if v >= q75:
            return "red"
        elif v >= q50:
            return "orange"
        elif v >= q25:
            return "green"
        else:
            return "blue"

    fg_name = layer_name or f"Latest {parameter}"
    fg = folium.FeatureGroup(name=fg_name, show=show)

    for _, row in df_latest.iterrows():
        value = float(row["value"])
        colour = colour_for_value(value)
        popup_html = (
            f"<b>Station:</b> {row['stationReference']}<br>"
            f"<b>{parameter.title()} Value:</b> {value}<br>"
            f"<b>Time:</b> {row['dateTime']}"
        )
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=6,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.8,
            popup=popup_html,
        ).add_to(fg)

    fg.add_to(fmap)

    return fmap


# ----------------------------------------------------------------------------------------
# Rainfall and Water Level Overlay Layer +++Combine ALL MAPS
# ----------------------------------------------------------------------------------------
def plot_combined_flood_and_rain_map(
    df_units: pd.DataFrame,
    df_rain: pd.DataFrame,
    *,
    la_col: str = "localAuthority",
    boundary_path: str | Path = None,
    boundary_name_col: str = "LAD23NM",
    risk_col: str = "riskLabel",
    postcode_col: str = "postcode",
    hist_col: str = "historicallyFlooded",
    zoom_start: int = 7,
    rain_parameter: str = "rainfall",
    rain_lat_col: str = "latitude",
    rain_lon_col: str = "longitude",
) -> folium.Map:
    """
    Combined map with:
    - Local authority polygons
    - Historically flooded postcode points coloured by flood risk
    - Latest rainfall (or level) as a separate toggleable layer
    """
    # --- 1. First create the LA + flood-risk map using your existing logic ---

    # This is basically an inline version of plot_local_authority_real_shapes,
    # slightly simplified so we can attach the rainfall layer at the end.

    if boundary_path is None:
        raise ValueError("You must provide boundary_path to the .gpkg file.")

    boundary_path = Path(boundary_path)
    if not boundary_path.exists():
        raise FileNotFoundError(f"Boundary file not found at: {boundary_path}")

    # Load boundaries
    gdf_bound = gpd.read_file(boundary_path)
    if gdf_bound.crs is not None and gdf_bound.crs.to_string() != "EPSG:4326":
        gdf_bound = gdf_bound.to_crs("EPSG:4326")

    # Work on a copy and keep only historically flooded rows
    df = df_units.copy()
    if hist_col not in df.columns:
        raise KeyError(f"Column '{hist_col}' not found in df_units.")
    df = df[df[hist_col] == 1].copy()
    if df.empty:
        raise ValueError(f"No rows with '{hist_col} == 1' – nothing to plot.")

    # Clean names for matching with boundaries
    df[la_col] = df[la_col].astype(str).str.strip()
    gdf_bound[boundary_name_col] = (
        gdf_bound[boundary_name_col].astype(str).str.strip()
    )

    # Filter boundaries to only those LAs present in flooded data
    las_predicted = df[la_col].unique()
    gdf_subset = gdf_bound[
        gdf_bound[boundary_name_col].isin(las_predicted)
    ].copy()
    if len(gdf_subset) == 0:
        raise ValueError(
            "No matching local authority names between model and boundary "
            "file."
        )

    # Colour polygons per authority
    colors = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "darkred",
        "lightred",
        "beige",
        "darkblue",
        "darkgreen",
        "cadetblue",
        "darkpurple",
        "white",
        "pink",
        "lightblue",
        "lightgreen",
        "gray",
        "black",
        "lightgray",
    ]
    la_list = sorted(gdf_subset[boundary_name_col].unique())
    color_map = {la: colors[i % len(colors)] for i, la in enumerate(la_list)}

    def style_function(feature):
        la_name = feature["properties"][boundary_name_col]
        col = color_map.get(la_name, "lightgray")
        return {
            "fillColor": col,
            "color": col,
            "weight": 1,
            "fillOpacity": 0.45,
        }

    # Base map centred on flooded postcodes
    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)

    # Add LA polygons
    folium.GeoJson(
        gdf_subset,
        name="Local Authorities",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=[boundary_name_col],
            aliases=["Local Authority:"],
        ),
    ).add_to(m)

    # Flood-risk points layer (only histFlood == 1)
    def risk_color(r):
        r = int(r)
        mapping = {
            0: "lightgray",
            1: "green",
            2: "yellow",
            3: "orange",
            4: "red",
            5: "darkred",
            6: "purple",
        }
        return mapping.get(r, "black")

    points_layer = folium.FeatureGroup(
        name="Flood risk points (hist=1)", show=True
    )

    for _, row in df.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        risk = row[risk_col]

        popup_parts = []
        if postcode_col in df.columns:
            popup_parts.append(f"Postcode: {row[postcode_col]}")
        popup_parts.append(f"Risk label: {risk}")
        popup_parts.append(f"Historically flooded: {row[hist_col]}")

        popup_html = "<br>".join(popup_parts)
        col = risk_color(risk)

        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color=col,
            fill=True,
            fill_color=col,
            fill_opacity=0.7,
            weight=0.3,
            popup=popup_html,
        ).add_to(points_layer)

    points_layer.add_to(m)

    # --- 2. Add latest rainfall (or level) as a separate layer on top ---

    m = add_latest_weather_layer(
        fmap=m,
        df=df_rain,
        parameter=rain_parameter,
        lat_col=rain_lat_col,
        lon_col=rain_lon_col,
        show=False,
        layer_name=f"Latest {rain_parameter}",
    )
    folium.LayerControl(collapsed=False).add_to(m)

    return m


def plot_color_size_map(
    df,
    feature,
    size_feature=None,
    lat_col="latitude",
    lon_col="longitude",
    invert_size=False,
):
    """
    Map plotting for two features: color-coded by "feature" and
    optionally marker size by "size_feature".

    Parameters
    ----------
    df: pd.DataFrame
        Must contain latitude and longitude columns for geographical plotting.
    feature: str
        Column to visualise with color.
    size_feature: str or None
        Column to visualise with marker radius. If None, uses fixed size.
    lat_col, lon_col: str
        GPS coordinate column names. Default "latitude" and "longitude".
    invert_size: bool
        If True, larger values of size_feature produce smaller markers.
        Useful for visualising e.g. distanceToWatercourse for risk analysis.

    Returns
    -------
    Folium map object.

    Example
    --------
    Plot riskLabel for color, medianPrice for marker size:
    >>> plot_color_size_map(df, feature="riskLabel",    # doctest: +SKIP
                            size_feature="medianPrice")
    """

    # Plot a folium map roughly centered around England
    m = folium.Map(location=[51.0, 0], zoom_start=7)

    color_values = df[feature]

    # Remove rows with NaN in latitude or longitude
    df = df.copy()
    df = df.dropna(subset=[lat_col, lon_col])

    color_scale = None  # initialize

    # Detect whether the feature is categorical
    if color_values.dtype == "object" or df[feature].nunique() <= 10:
        categories = sorted(color_values.dropna().unique())
        n = len(categories)

        if n == 1:

            def get_color(val):
                return cm.linear.OrRd_09.scale(0, 1)(0.5)

            step_colormap = None  # no legend needed
        else:
            # Create a StepColormap with discrete colors for each category
            step_colormap = cm.linear.OrRd_09.scale(
                min(categories), max(categories)
            ).to_step(n)
            step_colormap.index = list(categories)  # map steps

            # Map category value to color
            cat_to_color = {cat: step_colormap(cat) for cat in categories}

            def get_color(val):
                return cat_to_color[val]

    else:
        # Continuous feature
        vmin, vmax = np.percentile(color_values, [1, 99])
        color_scale = cm.linear.YlGnBu_09.scale(vmin, vmax)

        def get_color(val):
            return color_scale(val)

        step_colormap = None  # to handle legend later

    # Handle size_feature if provided
    if size_feature:
        size_values = df[size_feature]
        smin, smax = np.percentile(size_values.dropna(), [1, 99])

        # Add title
        map_title = f"Color: {feature}\nSize: {size_feature}"
        title_html = f"""<h1 style="position:absolute;z-index:100000;
        left:5vw;top:10px;font-size:16px;" >{map_title}</h1>"""
        m.get_root().html.add_child(folium.Element(title_html))

        def get_radius(val):
            if invert_size:
                return np.interp(val, [smin, smax], [25, 5])
            else:
                return np.interp(val, [smin, smax], [5, 25])

    else:
        # Add title
        map_title = f"Plot of {feature}"
        title_html = f"""<h1 style="position:absolute;z-index:100000;
        left:5vw;top:10px;font-size:16px;" >{map_title}</h1>"""
        m.get_root().html.add_child(folium.Element(title_html))

        # default radius
        def get_radius(val):
            return 6

    # Plot points
    for _, row in df.iterrows():
        cval = row[feature]
        sval = row[size_feature] if size_feature else None

        if pd.isna(cval):
            continue

        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=get_radius(sval),
            color=get_color(cval),
            popup=f"{feature}: {cval}"
            + (f"<br>{size_feature}: {sval}" if size_feature else ""),
        ).add_to(m)

    # Add legend if categorical or continuous
    if step_colormap is not None:
        step_colormap.caption = f"{feature} (categorical)"
        step_colormap.add_to(m)
    elif color_scale is not None:
        color_scale.caption = f"{feature} (continuous)"
        color_scale.add_to(m)

    return m


def filtered_plot(
    df,
    feature1,
    feature2=None,
    lat_col="latitude",
    lon_col="longitude",
    feature1_min=None,
    feature1_max=None,
    feature2_min=None,
    feature2_max=None,
    invert_size=False,
):
    """
    Map plotting for two features: color-coded by "riskLabel" and
    optional marker size by "medianPrice", with optional thresholds.
    The features must be numeric, including categorical labels (e.g. 1, 2, 3).

    Parameters
    ----------
    df: pd.DataFrame
        Must contain latitude and longitude columns.
    feature1: str
        A feature to visualise with color.
    feature2: str or None
        An optional feature to visualise with marker size.
    lat_col, lon_col: str
        GPS coordinate column names. Default "latitude", "longitude"
    feature1_min, feature1_max: float or None
        Optional: Min and max threshold for filtering the color
        feature (inclusive).
    feature2_min, feature2_max: float or None
        Optional: Min and max threshold for filtering the size
        feature (inclusive).
    invert_size: bool
        If True, larger values of feature2 produce smaller markers.
        Useful for visualising e.g. distanceToWatercourse for risk analysis.

    Returns
    -------
    Folium map object.

    Example
    --------
    Plot riskLabel for color, distanceToWatercourse for marker size (inverted),
    for riskLabel > 5, distanceToWatercourse >= 250m:
    >>> filtered_plot(df1, feature1="riskLabel",            # doctest: +SKIP
                      feature2="distanceToWatercourse",
                      feature1_min=6, feature1_max=None,
                      feature2_min=None, feature2_max=250, invert_size=True)
    """

    filtered_df = df.copy()
    if feature1_min is not None:
        filtered_df = filtered_df[filtered_df[feature1] >= feature1_min]
    if feature1_max is not None:
        filtered_df = filtered_df[filtered_df[feature1] <= feature1_max]
    if feature2:
        if feature2_min is not None:
            filtered_df = filtered_df[filtered_df[feature2] >= feature2_min]
        if feature2_max is not None:
            filtered_df = filtered_df[filtered_df[feature2] <= feature2_max]

    filtered_df = filtered_df.dropna()

    # Check if filtered dataframe is empty
    if filtered_df.empty:
        print("No data remaining after filtering.")
        return None

    return plot_color_size_map(
        filtered_df,
        feature1,
        feature2,
        lat_col=lat_col,
        lon_col=lon_col,
        invert_size=invert_size,
    )


def plot_latest_rainfall_and_level(
    df, lat_col="latitude", lon_col="longitude"
):
    """
    Plot the latest rainfall and river level measurements for each station
    on the same map, with separate color scales.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing columns: stationReference, dateTime, parameter,
        value, latitude, longitude.
    lat_col, lon_col : str
        Column names for latitude and longitude.

    Returns
    -------
    Folium map object
    """
    df = df.copy()
    df["dateTime"] = pd.to_datetime(df["dateTime"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Initialize map
    m = folium.Map(location=[51.0, 0], zoom_start=7)

    # Define parameters and their colormaps
    params = {"rainfall": cm.linear.YlOrRd_09, "level": cm.linear.Blues_09}

    for parameter, colormap_class in params.items():
        df_param = df[df["parameter"].str.contains(parameter, case=False)]
        if df_param.empty:
            continue

        # Pick latest measurement for each station
        df_latest = df_param.sort_values("dateTime").drop_duplicates(
            subset=["stationReference"], keep="last"
        )
        df_latest = df_latest.dropna(subset=[lat_col, lon_col, "value"])

        if df_latest.empty:
            continue

        # Colormap based on 5th-95th percentile
        vmin = df_latest["value"].quantile(0.05)
        vmax = df_latest["value"].quantile(0.95)
        colormap = colormap_class.scale(vmin, vmax)

        # Create feature group for toggle
        fg = folium.FeatureGroup(name=parameter.title())

        # Add markers
        for _, row in df_latest.iterrows():
            color = colormap(row["value"])
            folium.CircleMarker(
                location=[row[lat_col], row[lon_col]],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=(
                    f"<b>Station:</b> {row['stationReference']}<br>"
                    f"<b>{parameter.title()} Value:</b> {row['value']}<br>"
                    f"<b>Time:</b> {row['dateTime']}"
                ),
            ).add_to(fg)
        fg.add_to(m)
        colormap.caption = f"Latest {parameter.title()} Value"
        colormap.add_to(m)
    # Add layer control to toggle layers
    folium.LayerControl(collapsed=False).add_to(m)
    return m
