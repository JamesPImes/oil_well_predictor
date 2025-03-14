
"""
Calculate midpoints and distances for wells in bulk.
"""

import pandas as pd

from preprocess.shl_bhl.lat_long import (
    calc_lateral_distance,
    calc_lat_long_midpoint,
)

__all__ = [
    'DistanceCalculator',
    'get_distance_calculator',
    'calc_all_midpoints',
]


class DistanceCalculator:
    """Distance calculator with caching."""

    def __init__(self, locations: dict[str, tuple[float, float]]):
        """
        :param locations: A dict mapping API numbers to their locations
         (represented by the lat/long coord of their midpoints).
        """
        self.locations = locations
        self._coords_cache = {}
        self._well_cache = {}

    def calc_distance_between_coords(self, u: tuple, v: tuple):
        """
        Calculate distance between two lat/long coords. Cache the
        results.

        :param u: lat/long coords of the first point.
        :param v: lat/long coords of the second point.
        :return: distance between ``u`` and ``v`` in feet.
        """
        distance = self._coords_cache.get((u, v), None)
        if distance is None:
            distance = self._coords_cache.get((v, u), None)
        if distance is None:
            distance = calc_lateral_distance(u, v)
            self._coords_cache[(u, v)] = distance
        return distance

    def calc_distance_between_wells(self, api_num1: str, api_num2: str):
        """
        Calculate distance between two wells.
        :param api_num1: The API number of the first well.
        :param api_num2: The API number of the second well.
        :return: Distance in feet between the midpoints of the two
         wells.
        """
        dist = self._well_cache.get((api_num1, api_num2), None)
        if dist is not None:
            return dist
        midpoint_1 = self.locations[api_num1]
        midpoint_2 = self.locations[api_num2]
        dist = self.calc_distance_between_coords(midpoint_1, midpoint_2)
        self._well_cache[(api_num1, api_num2)] = dist
        self._well_cache[(api_num2, api_num1)] = dist
        return dist

    def calc_all_distances_from_well(self, api_num, avail_api_nums=None) -> dict[str, float]:
        """
        Calculate distances for all other wells, starting from this well.

        :param api_num: The API number of the target well.
        :param avail_api_nums: A collection of API numbers to
         avail_api_nums in the search. If not passed, will check
         everything in ``.locations`` (which could result in a distance
         of 0, if the target is in that dict).
        :return: Dict of well API numbers and their distance to the
         target well.
        """
        dists = {}
        if avail_api_nums is None:
            avail_api_nums = self.locations.keys()
        for other_api in avail_api_nums:
            if api_num == other_api:
                continue
            dists[other_api] = self.calc_distance_between_wells(api_num, other_api)
        return dists

    def calc_all_distances_from_location(self, location, avail_api_nums=None) -> dict[str, float]:
        """
        Calculate distances for all wells, starting from this location.

        :param location: The location of the target well (e.g., lat/long
         coord midpoint).
        :param avail_api_nums: A collection of API numbers to
         avail_api_nums in the search. If not passed, will check
         everything in ``.locations`` (which could result in a distance
         of 0, if the target is in that dict).
        :return: Dict of well API numbers and their distance to the
         target location.
        """
        dists = {}
        if avail_api_nums is None:
            avail_api_nums = self.locations.keys()
        for other_api in avail_api_nums:
            other_location = self.locations[other_api]
            dists[other_api] = self.calc_distance_between_coords(location, other_location)
        return dists

def calc_all_midpoints(
        wells: pd.DataFrame,
        lat_shl_header: str = 'lat_shl',
        long_shl_header: str = 'long_shl',
        lat_bhl_header: str = 'lat_bhl',
        long_bhl_header: str = 'long_bhl',
        api_header='API_Label'
) -> dict[str, tuple[float, float]]:
    """
    Calculate all midpoints for a set of wells.

    :param wells: Dataframe containing well data, including SHL / BHL
     data.
    :param lat_shl_header: Header string for the SHL latitude column.
    :param long_shl_header: Header string for the SHL longitude column.
    :param lat_bhl_header: Header string for the BHL latitude column.
    :param long_bhl_header: Header string for the BHL longitude column.
    :param api_header: Header string for the API number in the ``wells``
     dataframe.
    :return: Dict of each well's midpoint, keyed by its API number.
    """
    midpoints = {}
    for _, row in wells.iterrows():
        shl = row[lat_shl_header], row[long_shl_header]
        bhl = row[lat_bhl_header], row[long_bhl_header]
        midpoint = calc_lat_long_midpoint(shl, bhl)
        api = row[api_header]
        midpoints[api] = midpoint
    return midpoints

def get_distance_calculator(
    wells: pd.DataFrame,
    lat_shl_header: str = 'lat_shl',
    long_shl_header: str = 'long_shl',
    lat_bhl_header: str = 'lat_bhl',
    long_bhl_header: str = 'long_bhl',
    api_header = 'API_Label'
) -> DistanceCalculator:
    """
    Get a ``DistanceCalculator`` for the dataframe of ``wells``. Also
    calculates the midpoint each well to use as its location.

    :param wells: Dataframe containing well data, including SHL / BHL
     data.
    :param lat_shl_header: Header string for the SHL latitude column.
    :param long_shl_header: Header string for the SHL longitude column.
    :param lat_bhl_header: Header string for the BHL latitude column.
    :param long_bhl_header: Header string for the BHL longitude column.
    :param api_header: Header string for the API number in the ``wells``
     dataframe.
    :return: Dict of each well's midpoint, keyed by its API number.
    """
    midpoints = calc_all_midpoints(
        wells,
        lat_shl_header,
        long_shl_header,
        lat_bhl_header,
        long_bhl_header,
        api_header,
    )
    dist_calculator = DistanceCalculator(locations=midpoints)
    return dist_calculator
