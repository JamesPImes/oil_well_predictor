import pandas as pd

from .distance_calculator import get_distance_calculator, DistanceCalculator

__all__ = [
    'idw_weighting',
    'KNearestWellsFinder'
]


class KNearestWellsFinder:
    """
    Find the k-nearest wells to a target location or well.

    Note: Uses a cache for each unique value of ``k`` and the wells or
    locations searched, to help the tuning of (hyper)parameters other
    than ``k``.
    """

    def __init__(
            self,
            wells: pd.DataFrame,
            distance_calculator: DistanceCalculator = None):
        """
        :param wells: A dataframe of wells, including their locations.
        :param distance_calculator: A ``DistanceCalculator`` to use. If
         provided, it should be prepared with all of the wells in the
         ``wells`` dataframe. (The benefit of providing a prepared
         ``DistanceCalculator`` here is that it will use cached
         calculations where possible.)
        """
        self.wells = wells
        if distance_calculator is None:
            distance_calculator = get_distance_calculator(wells)
        self.distance_calculator = distance_calculator
        # Cache keyed by `k`.
        # Nest with another dict of API / location, value is nearest k wells and dists.
        self._cache = {}

    def _check_cache(self, k, search_term):
        """
        INTERNAL USE

        Check the cache if we've already found the nearest ``k`` wells
        for a given search term (either API number or location).
        """
        if k not in self._cache:
            self._cache[k] = {}
            return None
        return self._cache[k].get(search_term, None)

    def find_nearest_from_well(self, api_num, k) -> dict[str, float]:
        """
        Find the nearest ``k`` wells to the well represented by
        ``api_num`` (which must be in the ``.wells`` dataframe).
        :param api_num: API number of the target well.
        :param k: How many wells to select.
        :return: A dictionary of the nearest ``k`` wells and their
         distances to the target.
        """
        well_distances = self._check_cache(k, api_num)
        if well_distances is not None:
            return well_distances
        well_distances = self.distance_calculator.calc_all_distances_from_well(
            api_num, avail_api_nums=self.wells['API_Label'])
        nearest_k = self._reduce_to_nearest_k(well_distances, k)
        self._cache[k][api_num] = nearest_k
        return nearest_k

    def find_nearest_from_location(self, location, k) -> dict[str, float]:
        """
        Find the nearest ``k`` wells to the target ``location`` (a
        lat/long coord).
        :param location: The lat/long coord to search from (e.g., the
         midpoint of a target well).
        :param k: How many wells to select.
        :return: A dictionary of the nearest ``k`` wells and their
         distances to the target.
        """
        well_distances = self._check_cache(k, location)
        if well_distances is not None:
            return well_distances
        well_distances = self.distance_calculator.calc_all_distances_from_location(
            location, avail_api_nums=self.wells['API_Label'])
        nearest_k = self._reduce_to_nearest_k(well_distances, k)
        self._cache[k][location] = nearest_k
        return nearest_k

    def _reduce_to_nearest_k(self, distances, k):
        """
        INTERNAL USE

        Reduce a dict of ``{well: distance}`` pairs to the ``k`` nearest
        to the target.
        :param distances: A dict of all ``{well: distance}`` pairs.
        :param k: How many wells to select.
        :return: A dict of the ``k`` nearest ``{well: distance}`` pairs.
         (The resulting dict should be in ascending order of distance).
        """
        # Sort all by distance.
        well_dist_pairs = sorted(distances.items(), key=lambda item: item[1])
        nearest_k_as_tuples = well_dist_pairs[:k]
        # Convert tuples to dict.
        nearest_k = {api: dist for api, dist in nearest_k_as_tuples}
        return nearest_k


def idw_weighting(distances: dict, power: int = 2) -> dict:
    """
    Calculate inverse distance weighting (IDW) from Euclidean distances.
    Skew the weighting in favor of closer points by increasing ``power``
    (2 by default).

    Note: The weights are normalized to total 1.

    :param distances: A dict of ``{key: Euclidean distance}``. (Keys
     will be maintained in the returned dict.)
    :param power: Power parameter in the IDW weighting function. Higher
     values give higher weight to closer points. Default is 2.
    :return: A dict of ``{key: weight}``, where keys are the same as in
     the original dict.
    """
    idw = {k: (1 / v)**power for k, v in distances.items()}
    sum_weights = sum(idw.values())
    norm_weights = {k: (v / sum_weights) for k, v in idw.items()}
    return norm_weights
