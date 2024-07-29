from calendar import monthrange
from datetime import datetime

import pandas as pd
import numpy as np

__all__ = [
    'ProductionPreprocessor',
    'get_days_in_month',
    'first_day_of_month',
    'last_day_of_month',
]


class ProductionPreprocessor:
    def __init__(
            self,
            prod_records: pd.DataFrame,
            bbls_col: str = 'Oil Produced',
            days_prod_col: str = 'Days Produced',
            date_col: str = 'First of Month',
            formation_col: str = 'Formation',
    ):
        """
        :param prod_records: A dataframe of monthly production records.
        :param bbls_col: Header for BBLs produced each month.
        :param days_prod_col: Header for number of days produced each month.
        :param date_col: Header for the date column.
        """
        self.df = prod_records
        self.bbls_col = bbls_col
        self.days_prod_col = days_prod_col
        self.date_col = date_col
        self.formation_col = formation_col
        self.prod_cols = [
            self.bbls_col,
        ]

    def preprocess_all(self) -> pd.DataFrame:
        self.fill_missing_months()
        self.drop_leading_nonproducing_months()
        self.drop_first_incomplete_month()
        self.clean_formation()
        self.recategorize_not_completed()
        self.zero_out_negatives()
        self.fill_prod_per_day()
        return self.df

    @property
    def first_month(self) -> datetime:
        """Get the first day of the first month as a ``datetime``."""
        first = self.df[self.date_col].min()
        return first_day_of_month(first)

    @property
    def last_month(self) -> datetime:
        """Get the first day of the last month as a ``datetime``."""
        last = self.df[self.date_col].max()
        return first_day_of_month(last)

    def fill_missing_months(self) -> pd.DataFrame:
        """
        Convert all dates in the configured ``.date_col`` to the first
        of the month, fill in any missing months between the first and
        last months, and sort by date (ascending). Any added dates will
        have values of 0 for the relevant fields.

        Store the results to ``.df`` as a deep copy of the original.
        """
        df = self.df.copy(deep=True)
        df[self.date_col] = df[self.date_col].apply(lambda x: first_day_of_month(x))

        # Ensure there are no months missing from the data.
        every_month = pd.DataFrame()
        every_month[self.date_col] = pd.date_range(
            start=self.first_month, end=self.last_month, freq='MS')
        df = df.merge(every_month, how='outer', on=self.date_col).fillna(0)
        df = df.sort_values(by=[self.date_col], ascending=True)
        self.df = df
        return df

    def drop_leading_nonproducing_months(self) -> pd.DataFrame:
        """
        Drop all monthly production records before the first month
        during which production actually occurred.
        :return:
        """
        self.fill_missing_months()
        sum_prod = 0
        while sum_prod == 0:
            first = self.first_month
            sum_prod = 0
            for prod_col in self.prod_cols:
                sum_prod += self.df[self.df[self.date_col] == first][prod_col].sum()
            if sum_prod == 0:
                self.df = self.df.drop(self.df[self.df[self.date_col] == first].index)
        self.df = self.df
        return self.df

    def drop_first_incomplete_month(self) -> pd.DataFrame:
        """
        Drop the first month if it did not produce every day of the month.
        :return:
        """
        df = self.df
        first_month = self.first_month
        days_produced = df[df[self.date_col] == first_month][self.days_prod_col].max()
        calendar_days = get_days_in_month(first_month)
        if days_produced != calendar_days:
            self.df = df.drop(df[df[self.date_col] == first_month].index)
        return self.df

    def clean_formation(self) -> pd.DataFrame:
        """Trim white space around the formation names."""
        self.df[self.formation_col] = self.df[self.formation_col].str.strip()
        return self.df

    def recategorize_not_completed(self) -> pd.DataFrame:
        """
        For wells that initially reported "Not Completed" as the
        formation but later report a specific formation, recategorize
        those earlier months as that formation.
        :return:
        """
        nc_formations = [
            'NOT COMPLETED',
        ]
        self.clean_formation()
        formations = self.df[self.formation_col].unique()
        meaningful_formations = set(formations) - set(nc_formations)
        if len(meaningful_formations) == 1 and len(formations) > 1:
            formation = meaningful_formations.pop()
            self.df[self.formation_col] = self.df[self.formation_col].replace(nc_formations, formation)
        return self.df

    def zero_out_negatives(self) -> pd.DataFrame:
        """
        Replace negative values in the production column(s) with zero.
        :return:
        """
        df = self.df
        for prod_col in self.prod_cols:
            df[prod_col] = np.where(df[prod_col] < 0, 0, df[prod_col])
        self.df = df
        return df

    def fill_prod_per_day(self) -> pd.DataFrame:
        """
        Calculates and fills BBLs produced per production-day
        (``'bbls_per_prod_day'``) and BBLs produced per calendar day
        (``'bbls_per_calendar_day'``).  Also fills ``calendar_days``.

        Note: Does not consider multiple producing formations within a given
        well to occur during the same month, but rather treats each row as
        a separate observation.
        """
        df = self.df
        df[self.bbls_col] = df[self.bbls_col].fillna(value=0)
        df['bbls_per_prod_day'] = df[self.bbls_col] / df[self.days_prod_col]
        df['bbls_per_prod_day'] = df['bbls_per_prod_day'].fillna(value=0)
        df['calendar_days'] = df[self.date_col].apply(get_days_in_month)
        df['bbls_per_calendar_day'] = df[self.bbls_col] / df['calendar_days']
        self.df = df
        return df


def get_days_in_month(dt) -> int:
    """
    From a ``datetime`` object, get the total number of days in a
    given calendar month.

    :param dt: A ``datetime`` object.
    :return: The number of days in the month.
    """
    _, last_day = monthrange(dt.year, dt.month)
    return last_day


def first_day_of_month(dt: datetime) -> datetime:
    """
    Get the date of the first day of the month from the ``datetime``.
    """
    return datetime(dt.year, dt.month, 1)


def last_day_of_month(dt):
    """
    Get the date of the last day of the month from the ``datetime``.
    """
    last_day = get_days_in_month(dt)
    return datetime(dt.year, dt.month, last_day)
