*******************************
Visualising HistoricFlooding & FloodRisk with Local Authority
*******************************

This module provides helper functions for visualising flood-related data
using Folium. It **does not** train any models. Instead, it expects you to
pass in data (for example, a :class:`pandas.DataFrame` with latitude, longitude,
and model predictions).

Typical usage is from ``analysis.py`` or directly in a Jupyter notebook.

.. note::

   The examples below assume the module is available as
   ``flood_tool.visualisation.mapping``. If your path is different,
   update the import lines accordingly.

Quick start
-----------------------------

Base map from a single point::

    from flood_tool.visualisation import mapping as ftmap

    m = ftmap.make_base_map(centre_lat=51.5, centre_lon=-0.1, zoom_start=10)
    m

Base map from a DataFrame with latitude/longitude columns::

    m = ftmap.make_interactive_base_map(df, lat_col="latitude", lon_col="longitude")
    m

Plot historic flooding for a single postcode::

    m = ftmap.plot_historic_flood_for_postcode("BH12 1AA")
    m

Plot historic flooding grouped by postcode area (BA, BN, BH, …)::

    m = ftmap.plot_flood_by_area(df_postcodes)
    m

Overlay latest rainfall data on an existing map::

    m = ftmap.plot_flood_by_area(df_postcodes)
    m = ftmap.add_latest_weather_layer(m, df_weather, parameter="rainfall")
    m

Module reference
-----------------------------

.. automodule:: flood_tool.visualisation.mapping
   :noindex:

Base map helpers
-----------------------------

make_base_map
-----------------------------

.. autofunction:: flood_tool.visualisation.mapping.make_base_map

make_interactive_base_map
-----------------------------

.. autofunction:: flood_tool.visualisation.mapping.make_interactive_base_map

_make_colour_lookup
-----------------------------

.. autofunction:: flood_tool.visualisation.mapping._make_colour_lookup


Latest weather plotting
-----------------------

plot_latest_weather_data
-----------------------------

.. autofunction:: flood_tool.visualisation.mapping.plot_latest_weather_data

add_latest_weather_layer
-----------------------------

.. autofunction:: flood_tool.visualisation.mapping.add_latest_weather_layer


Historic flood classifier (postcode)
------------------------------------

plot_historic_flood_for_postcode
-----------------------------

.. autofunction:: flood_tool.visualisation.mapping.plot_historic_flood_for_postcode

plot_flood_by_area
-----------------------------

.. autofunction:: flood_tool.visualisation.mapping.plot_flood_by_area


Local authority visualisation
-----------------------------

plot_local_authority_map
-----------------------------

.. autofunction:: flood_tool.visualisation.mapping.plot_local_authority_map

plot_local_authority_real_shapes
-----------------------------
.. autofunction:: flood_tool.visualisation.mapping.plot_local_authority_real_shapes
