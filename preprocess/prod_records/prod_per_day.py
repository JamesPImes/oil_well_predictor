from calendar import monthrange

import pandas as pd

__all__ = [
    'fill_prod_per_day',
]


def fill_prod_per_day(
        prod_records: pd.DataFrame,
        bbls_col: str = 'Oil Produced',
        days_prod_col: str = 'Days Produced',
        date_col: str = 'First of Month'
) -> pd.DataFrame:
    """
    Calculates and fills BBLs produced per production-day (``'bbls_per_prod_day'``)
    and BBLs produced per calendar day (``'bbls_per_calendar_day'``).
    Also fills ``calendar_days``.

    Note: Does not consider multiple producing formations within a given
    well to occur during the same month, but rather treats each row as
    a separate observation.

    :param prod_records: A dataframe of monthly production records.
    :param bbls_col: Header for BBLs produced each month.
    :param days_prod_col: Header for number of days produced each month.
    :param date_col: Header for the date column.
    :return:
    """
    df = prod_records
    df[bbls_col] = df[bbls_col].fillna(value=0)
    df['bbls_per_prod_day'] = df[bbls_col] / df[days_prod_col]
    df['bbls_per_prod_day'] = df['bbls_per_prod_day'].fillna(value=0)
    # monthrange() gives (first day, last day), and we only need the second val.
    df['calendar_days'] = df[date_col].apply(lambda dt: monthrange(dt.year, dt.month)[1])
    df['bbls_per_calendar_day'] = df[bbls_col] / df['calendar_days']
    return df
