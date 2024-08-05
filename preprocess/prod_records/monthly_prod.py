from calendar import monthrange
from datetime import datetime, timedelta
from functools import cached_property

import pandas as pd
import numpy as np

__all__ = [
    'ProductionPreprocessor',
    'get_days_in_month',
    'first_day_of_month',
    'last_day_of_month',
    'get_monthly_dates',
    'fill_calendar_days'
]


class ProductionPreprocessor:
    def __init__(
            self,
            prod_records: pd.DataFrame,
            bbls_col: str = 'Oil Produced',
            days_prod_col: str = 'Days Produced',
            date_col: str = 'First of Month',
            formation_col: str = 'Formation',
            lateral_length_ft: float = None
    ):
        """
        :param prod_records: A dataframe of monthly production records.
        :param bbls_col: Header for BBLs produced each month.
        :param days_prod_col: Header for number of days produced each month.
        :param date_col: Header for the date column.
        param: lateral_length_ft: The length of the lateral in feet.
        """
        self.df = prod_records
        self.bbls_col = bbls_col
        self.days_prod_col = days_prod_col
        self.date_col = date_col
        self.formation_col = formation_col
        self.lateral_length_ft = lateral_length_ft
        self.prod_cols = [
            self.bbls_col,
        ]
        self.formations = []

    def preprocess_all(self) -> pd.DataFrame:
        self.fill_missing_months()
        if not self.has_produced:
            self.drop_all()
            return self.df
        self.drop_leading_nonproducing_months()
        self.drop_first_incomplete_month()
        self.clean_formation()
        self.recategorize_not_completed()
        self.identify_formations()
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

    @cached_property
    def has_produced(self) -> bool:
        """
        Check if the well has produced anything.
        :return:
        """
        sum_prod = 0
        for prod_col in self.prod_cols:
            sum_prod += self.df[prod_col].sum()
        return sum_prod != 0

    def drop_all(self) -> pd.DataFrame:
        """
        Drop every row in the DataFrame.
        :return:
        """
        self.df = self.df.drop(self.df.index)
        return self.df

    def drop_leading_nonproducing_months(self) -> pd.DataFrame:
        """
        Drop all monthly production records before the first month
        during which production actually occurred.
        :return:
        """
        self.fill_missing_months()
        sum_prod = 0
        while sum_prod == 0 and len(self.df) > 0:
            first = self.first_month
            sum_prod = 0
            for prod_col in self.prod_cols:
                sum_prod += self.df[self.df[self.date_col] == first][prod_col].sum()
            if sum_prod == 0:
                self.df = self.df.drop(self.df[self.df[self.date_col] == first].index)
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

    def identify_formations(self) -> list[str]:
        """
        Identify the unique formations that appear in the production
        records. Store them to ``.formations`` attribute and return
        them.
        :return: A list of unique formations.
        """
        formations = [
            fm for fm in self.df[self.formation_col].unique()
            if not pd.isna(fm)
        ]
        self.formations = formations
        return formations

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

    def normalize_for_lateral(self, lateral_length_ft: float) -> pd.DataFrame:
        """
        Normalize the monthly production values into BBLs/ft (i.e., BBLs
        produced per foot of lateral).
        :param lateral_length_ft: The length of the lateral leg of this
         well.
        :return:
        """
        df = self.df
        df['bbls_per_prod_day_latnorm'] = df['bbls_per_prod_day'] / lateral_length_ft
        df['bbls_per_calendar_day_latnorm'] = df['bbls_per_calendar_day'] / lateral_length_ft
        self.df = df
        return df

    def mean_smooth_zero_production(self) -> pd.DataFrame:
        """
        EXPERIMENTAL

        Replace zero-production months with an average of the months
        immediately before and after that period of non-production
        (including those positive-production months). That is, for a
        period of no production from March through July (i.e., 4
        months), take the average monthly production from February
        through August (i.e., 6 months), and apply it to all 6 of those
        months.
        :return:
        """
        df = self.df
        # Sort by formation and date_col and reset indexes, so that we can
        # find start/end points (indexes) of periods of zero production.
        df = df.sort_values(by=[self.formation_col, self.date_col], ascending=True)
        df = df.reset_index(drop=True)
        fmn_col = self.formation_col
        formations = df[fmn_col].unique()
        for fmn in formations:
            for prod_col in self.prod_cols:
                # We will limit the values we replace to this particular
                # formation. Otherwise, we might consider
                fmn_idxs = df[df[fmn_col] == fmn].index
                start_fmn = min(fmn_idxs)
                end_fmn = max(fmn_idxs)
                zero_idxs = df[df[fmn_col] == fmn & df[prod_col] == 0].index
                zero_ranges = find_ranges(zero_idxs)
                for i, j in zero_ranges:
                    idx_i = max(i - 1, start_fmn)
                    idx_j = min(j + 1, end_fmn)
                    prod_a = df.iloc[[idx_i]][prod_col]
                    prod_b = df.iloc[[idx_j]][prod_col]
                    per_month = (prod_a + prod_b) / (idx_j - idx_i + 1)
                    df.loc[idx_i:idx_j, prod_col] = per_month
        df = df.sort_values(by=[self.date_col], ascending=True)
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


def find_ranges(nums: list) -> list:
    """
    Find ranges of consecutive integers in the set. Returns a list of
    tuples, of the first and last numbers (inclusive) in each sequence.

    :param nums: A collection of unique integers.
    :return: A list of 2-tuples of integers, being the min and max
     of each range (inclusive).
    """
    nnums = sorted(set(nums))
    starts = [n for n in nnums if n - 1 not in nums]
    ends = [n for n in nnums if n + 1 not in nums]
    return [*zip(starts, ends)]


def get_monthly_dates(start_date: datetime, days: int = 1460, date_col='First of Month') -> pd.DataFrame:
    """
    Get a new dataframe with monthly dates, beginning at the specified
    ``start_date`` and ending after the specified number of ``days``.
    The dataframe will also include the number of days in each month.
    :param start_date: The datetime at which to start monthly dates.
    :param days: The number of days to stop after.
    :param date_col: The name of the column to put dates in.
    :return: A dataframe with monthly dates and number of days in each.
    """
    df = pd.DataFrame()
    end_date = start_date + timedelta(days=days)
    df[date_col] = pd.date_range(start_date, end_date, freq='MS')
    df = fill_calendar_days(df, date_col)
    return df


def fill_calendar_days(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Create and fill a ``'calendar_days'`` column in a dataframe.
    :param df: A dataframe with monthly dates.
    :param date_col: The name of the column that contains dates.
    :return: The same dataframe, with ``'calendar_days'`` filled.
    """
    df['calendar_days'] = df[date_col].apply(get_days_in_month)
    return df
