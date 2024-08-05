import pandas as pd

from preprocess.shl_bhl.lat_long import (
    calc_lat_long_midpoint,
    calc_lateral_distance,
)

__all__ = [
    'find_k_nearest',
    'idw_weighting',
]


def find_k_nearest(
        wells: pd.DataFrame,
        target_shl: tuple[float, float],
        target_bhl: tuple[float, float],
        k: int,
) -> dict[int, float]:
    """
    Find k nearest wells and their respective distances, measured from
    the midpoints of their respective laterals.
    :param wells: A dataframe with SHL and BHL data for wells.
    :param target_shl: Lat/long tuple for SHL of target well.
    :param target_bhl: Lat/long tuple for BHL of target well.
    :param k: The number of neighbors to find.
    :return:
    """
    # TODO: Add similarity constraints (formation, lateral length).
    df = wells.copy(deep=True)
    target_midpoint = calc_lat_long_midpoint(shl_coord=target_shl, bhl_coord=target_bhl)
    midpt_distances = []
    for _, row in df.iterrows():
        dist = calc_lateral_distance(
            shl_coord=(row['lat_midpoint'], row['long_midpoint']),
            bhl_coord=target_midpoint,
        )
        midpt_distances.append(dist)
    df['distance_from_target'] = midpt_distances
    df = df.sort_values(by='distance_from_target', ascending=True)
    df.reset_index()
    top_k = df.iloc[:k]
    distances = {
        api: dist for api, dist in zip(top_k['API_Label'], top_k['distance_from_target'])
    }
    return distances


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
