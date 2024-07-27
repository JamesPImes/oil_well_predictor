
"""
Utilities for calculating midpoint and distance between lat/long coords.
"""

import pandas as pd
from geopy.distance import geodesic

__all__ = [
    'calc_lat_long_midpoint',
    'calc_lateral_distance',
    'fill_lat_long_midpoints',
    'fill_lateral_distances',
]


def calc_lat_long_midpoint(shl_coord, bhl_coord) -> (float, float):
    """
    Calculate the midpoint between the two lat/long coords.
    :param shl_coord:
    :param bhl_coord:
    :return: Latitude and longitude of the midpoint, as a 2-tuple.
    """
    lat = (shl_coord[0] + bhl_coord[0]) / 2
    long = (shl_coord[1] + bhl_coord[1]) / 2
    return (lat, long)


def calc_lateral_distance(shl_coord, bhl_coord) -> float:
    """
    Calculate the distance (in feet) between the two coordinates,
    assuming the lateral runs in a straight line.

    :param shl_coord: (lat, long) for SHL.
    :param bhl_coord: (lat, long) for BHL.
    :return: The distance in feet.
    """
    return geodesic(shl_coord, bhl_coord).ft


def fill_lat_long_midpoints(
        wells: pd.DataFrame,
        lat_shl: str = 'lat_shl',
        long_shl: str = 'long_shl',
        lat_bhl: str = 'lat_bhl',
        long_bhl: str = 'long_bhl',
) -> pd.DataFrame:
    """
    Create and fill columns ``'lat_midpoint'`` and ``'long_midpoint'``
    for the midpoint between the lat/long of the SHL and BHL. Columns
    are added to the original dataframe.

    :param wells: A dataframe of well data (SHL and BHL).
    :param lat_shl: Header name for SHL latitude.
    :param long_shl: Header name for SHL longitude.
    :param lat_bhl: Header name for BHL latitude.
    :param long_bhl:  Header name for BHL longitude.
    :return: The same dataframe, with the new columns.
    """
    wells['lat_midpoint'], wells['long_midpoint'] = calc_lat_long_midpoint(
        shl_coord=(wells[lat_shl], wells[long_shl]),
        bhl_coord=(wells[lat_bhl], wells[long_bhl]),
    )
    return wells


def fill_lateral_distances(
        wells: pd.DataFrame,
        lat_shl: str = 'lat_shl',
        long_shl: str = 'long_shl',
        lat_bhl: str = 'lat_bhl',
        long_bhl: str = 'long_bhl',
) -> pd.DataFrame:
    """
    Create and fill column ``'lateral_length_ft'``for the length of the
    lateral (in feet) between the SHL and BHL. Column is added to the
    original dataframe.

    :param wells: A dataframe of well data (SHL and BHL).
    :param lat_shl: Header name for SHL latitude.
    :param long_shl: Header name for SHL longitude.
    :param lat_bhl: Header name for BHL latitude.
    :param long_bhl:  Header name for BHL longitude.
    :return: The same dataframe, with the new column.
    """
    lateral_distances = []
    for _, row in wells.iterrows():
        lat_dist = calc_lateral_distance(
            shl_coord=(row[lat_shl], row[long_shl]),
            bhl_coord=(row[lat_bhl], row[long_bhl]),
        )
        lateral_distances.append(lat_dist)
    wells['lateral_length_ft'] = lateral_distances
    return wells
