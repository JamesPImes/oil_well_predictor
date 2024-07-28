import pandas as pd
import numpy as np

__all__ = [
    'ExpRegressionModel',
]


class ExpRegressionModel:
    """
    A weighted exponential regression model for monthly production
    records.
    """
    def __init__(
            self,
            a: float = np.nan,
            b: float = np.nan,
            day_ranges: tuple[int, int] = (0, 1461)):
        """
        We convert the exponential to a linear function:

        i.e.:  ``f(x) = a * e^(bx)  ==>  ln[f(x)] ==>  ln(a) + bx``

        ``a`` and ``b`` are the coefficients of that linear function,
        which we will later convert back in order to make predictions:
        ``f(x) = e^(a + bx)``
        ... or equivalently:
        ``f(x) = e^a * e^(bx)``

        (Simply use ``.predict()`` method after training).

        :param a:
        :param b:
        :param day_ranges:
        """
        self.a = a
        self.b = b
        self.day_ranges = day_ranges

    def train(self, prod_records: pd.DataFrame, day_ranges: tuple[int, int] = None):
        """
        Run a weighted exponential regression on monthly production
        records, limited to the specified ``day_ranges`` (defaults to
        the first 1461, i.e., the first 4 years).
        
        Resulting coefficients ``a`` and ``b`` are stored as attributes.
        Later, generate predictions from this regression with
        ``.predict()``.
        
        :param prod_records: A dataframe of monthly production records.
         Must already be preprocessed and sorted by month (ascending).
        :param day_ranges: Limit our review to the selected day ranges.
         Default is ``(0, 1461)`` -- i.e., the first 4 years.
        :return: None
        """
        if day_ranges is None:
            day_ranges = self.day_ranges
        prod_records = prod_records.copy(deep=True)
        # Swap 0's for arbitrarily small values to avoid log issues.
        prod_records['bbls_per_prod_day'] = prod_records['bbls_per_prod_day'].replace(0, 0.0000001)
        prod_records['bbls_per_calendar_day'] = prod_records['bbls_per_calendar_day'].replace(0, 0.0000001)
        x = prod_records['calendar_days'].cumsum()
        x = x.loc[lambda z: (z >= day_ranges[0]) & (z <= day_ranges[1])]
        y = prod_records['bbls_per_calendar_day'][x.index.values]

        # Weighted to favor larger numbers.
        poly_coefs = np.polyfit(x, np.log(y), 1, w=np.sqrt(y))
        self.b, self.a = poly_coefs
        return self.a, self.b

    def predict(self, cumulative_elapsed_days):
        """
        Generate predicted BBLs/day over time for the provided list of
        cumulative elapsed days (i.e. sum of calendar days for
        consecutive months), based on the ``a`` and ``b`` values from
        the exponential regression.

        :param cumulative_elapsed_days: A sequence of integers
         representing the total time passed since production began.
        :return: A sequence of floats, being the predicted BBLs/day that
         are produced.
        """
        return np.exp(self.a + self.b * cumulative_elapsed_days)

    @staticmethod
    def weight_models(models: list['ExpRegressionModel'], weights: list[float] = None) -> 'ExpRegressionModel':
        """
        Generate a new model by weighting multiple models.
        :param models: A list of previously trained ``ExpRegressionModel``
         objects.
        :param weights: A list of weights to use for the provided
         models. Must be equal in length to the list of models. If not
         provided, will give each model equal weighting.
        :return: A new ``ExpRegressionModel``.
        """
        n = len(models)
        if weights is None:
            weights = [1 / n] * n
        if len(weights) != n:
            raise ValueError('`models` and `weights` must have the same length')
        if sum(weights) != 1.0:
            raise ValueError('Sum of `weights` must equal 1.0')
        a = 0.0
        b = 0.0
        for model, weight in zip(models, weights):
            a += model.a * weight
            b += model.b * weight
        return ExpRegressionModel(a=a, b=b)
