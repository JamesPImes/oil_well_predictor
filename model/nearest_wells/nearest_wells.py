import pandas as pd

from preprocess.shl_bhl.lat_long import (
    calc_lat_long_midpoint,
    calc_lateral_distance,
)


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
