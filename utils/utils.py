import pandas as pd

__all__ = [
    'get_prod_window',
    'get_cumulative_days',
    'get_cumulative_production',
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

def get_cumulative_production(
        prod_records: pd.DataFrame,
        predicted_bbls: pd.Series = None,
        month_checkpoints=(24, 36, 48),
) -> (dict, dict):
    """
    Get cumulative produced BBLs (both actual and predicted) at any
    number of time frames (e.g., first 24 months, 36 months, 48 months).

    :param prod_records: A dataframe of preprocessed actual monthly
     production.
    :param predicted_bbls: A series of predicted daily BBLs (i.e., the
     output of a prediction model). Should be at least the same length
     of time as covered by ``prod_records`` (which is actual monthly
     production).
    :param month_checkpoints: A list of months to extract. Defaults to
     the first  24, 36, and 48 months. (Will also always include the
     maximum number of months in the ``prod_records`` dataframe.)
    :return: Two dicts, the first being the total actual production at
     the end of each requested period; the second being the total
     predicted production at the end of each requested period. The final
     month will also be included in each dict. If the requested month is
     greater than the months in the ``prod_records`` dataframe, then
     that entry in each returned dict will have a ``None`` value.
    """
    month_checkpoints = list(month_checkpoints)
    total_months = len(prod_records)
    if total_months not in month_checkpoints:
        month_checkpoints.append(total_months)
    actual = {}
    pred = {}
    for n_months in month_checkpoints:
        if n_months > total_months:
            actual[n_months] = None
            pred[n_months] = None
            continue
        actual_subset = prod_records.iloc[:n_months]
        cumul_days = sum(actual_subset['calendar_days'])
        actual[n_months] = sum(actual_subset['Oil Produced'])
        if predicted_bbls is not None:
            pred_subset = predicted_bbls.iloc[:cumul_days]
            pred[n_months] = sum(pred_subset)
    return actual, pred
