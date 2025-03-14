import pandas as pd

__all__ = [
    'get_prod_window',
    'get_cumulative_days',
]


def get_prod_window(
        prod_records: pd.DataFrame,
        min_months: int = 36,
        max_months: int = 48,
        discard_gaps: bool = True
) -> None | pd.DataFrame:
    """
    Pare the production records down to the specified number of months
    (min and max). If there are fewer than the specified number of
    ``min_months`` of production, will return ``None``. Otherwise, will
    return a new dataframe containing the production records.
    :param prod_records: A fully preprocessed dataframe of monthly
     production records for one well. (Must be sorted with all of the
     usual columns).
    :param min_months: The fewest number of months of production to
     accept. If there are not enough, this will return ``None``.
    :param max_months: The max number of months of production to
     include.
    :param discard_gaps: Whether to discard or keep months during which
     no production occurred. Defaults to ``True`` (i.e., discard).
    :return: A dataframe containing between ``min_months`` and
     ``max_months`` number of rows. If not enough months of production
     existed in the records, will return ``None``.
    """
    prod_records = prod_records.copy(deep=True)
    if discard_gaps:
        try:
            prod_records = prod_records.drop(prod_records[prod_records['bbls_per_calendar_day'] == 0].index)
            prod_records = prod_records.drop(prod_records[prod_records['bbls_per_calendar_day'].isna()].index)
        except KeyError:
            pass
    prod_records = prod_records.reset_index(drop=True)
    if len(prod_records) < min_months:
        # Not enough data.
        return None
    max_idx = min(max_months, len(prod_records))  # 1-indexed
    selected = prod_records.iloc[:max_idx]
    return selected.reset_index(drop=True)

def get_cumulative_days(prod_records: pd.DataFrame) -> pd.Series:
    """
    Get the cumulative days from the production records.

    :param prod_records: A fully preprocessed dataframe of monthly
     production records for one well. (Presumably culled to the target
     window for training or prediction.)
    :return: A Series of ``[1, n]``, where ``n`` is the total days
     represented in the ``prod_records``.
    """
    monthly_days = prod_records['calendar_days']
    return pd.Series(range(1, sum(monthly_days) + 1))
