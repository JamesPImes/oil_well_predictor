import pandas as pd
import numpy as np

__all__ = [
    'ExpRegressionModel',
]


class ExpRegressionModel:
    """
    A weighted exponential regression model for monthly production
    records (in BBLs/calendar day), adjusted for length of the lateral
    leg of the well.
    """

    def __init__(
            self,
            a: float = np.nan,
            b: float = np.nan,
            lateral_length_ft: float = None,
            day_ranges: tuple[int, int] = (0, 1461)):
        """
        We convert the exponential to a linear function:

        i.e.:  ``f(x) = A * e^(bx)  ==>  ln[f(x)] ==>  ln(A) + bx``
        where ``a = ln(A)``.

        The linear function will later be converted back in order to
        make predictions:  ``f(x) = e^(a + bx)``

        or equivalently:  ``f(x) = e^a * e^(bx)``

        (Use ``.predict_bbls_per_calendar_day()`` method after training.)

        :param a: Linear coefficient ``a``, as defined above. (If not
         defined here, it will be defined by calling ``.train()`` on a
         set of monthly production records.)
        :param b: Linear coefficient ``b``, as defined above. (If not
         defined here, it will be defined by calling ``.train()`` on a
         set of monthly production records.)
        :param: lateral_length_ft: The length of the lateral in feet.
         (If not specified here, must specify at ``.train()``.)
        :param day_ranges: Limit our review to the selected day ranges.
         Default is ``(0, 1461)`` -- i.e., the first 4 years.
        """
        self.a = a
        self.b = b
        self.day_ranges = day_ranges
        self.lateral_length_ft = lateral_length_ft

    def train(
            self,
            prod_records: pd.DataFrame,
            lateral_length_ft: float = None,
            day_ranges: tuple[int, int] = None,
    ) -> (float, float):
        """
        Run a weighted exponential regression on monthly production
        records, limited to the specified ``day_ranges`` (defaults to
        the first 1461, i.e., the first 4 years).

        Resulting coefficients ``a`` and ``b`` are stored as attributes.
        Later, generate predictions from this regression with
        ``.predict_bbls_per_calendar_day()``.

        :param prod_records: A dataframe of monthly production records.
         Must already be preprocessed and sorted by month (ascending).
        :param lateral_length_ft: The length of the lateral in feet.
         If not specified here, will pull from what was specified at
         init. If not specified there either, will raise a
         ``ValueError``. (If specified as this parameter, will also
         store to ``.lateral_length_ft`` attribute.)
        :param day_ranges: Limit our review to the selected day ranges.
         Default is ``(0, 1461)`` -- i.e., the first 4 years.
        :return: None
        """
        if day_ranges is None:
            day_ranges = self.day_ranges
        if lateral_length_ft is None:
            lateral_length_ft = self.lateral_length_ft
        else:
            self.lateral_length_ft = lateral_length_ft
        if lateral_length_ft is None:
            raise ValueError('Must specify lateral length.')
        prod_records = prod_records.copy(deep=True)
        # Swap 0's for arbitrarily small values to avoid log issues.
        prod_records['bbls_per_prod_day'] = prod_records['bbls_per_prod_day'].replace(0, 0.0000001)
        prod_records['bbls_per_calendar_day'] = prod_records['bbls_per_calendar_day'].replace(0, 0.0000001)
        x = prod_records['calendar_days'].cumsum()
        x = x.loc[lambda z: (z >= day_ranges[0]) & (z <= day_ranges[1])]
        y = prod_records['bbls_per_calendar_day'][x.index.values] / lateral_length_ft

        # Weighted to favor larger numbers.
        poly_coefs = np.polyfit(x, np.log(y), 1, w=np.sqrt(y))
        self.b, self.a = poly_coefs
        return self.a, self.b

    def predict_bbls_per_calendar_day(self, cumulative_elapsed_days, lateral_length_ft: float = None):
        """
        Generate predicted BBLs/day over time for the provided list of
        cumulative elapsed days (i.e. sum of calendar days for
        consecutive months), based on the ``a`` and ``b`` values from
        the exponential regression.

        :param cumulative_elapsed_days: A sequence of integers
         representing the total time passed since production began.
        :param lateral_length_ft: The length of the lateral leg of the
         candidate well. If not provided, will assume that the input
         used to train the model was NOT normalized for lateral length.
        :return: A sequence of floats, being the predicted BBLs/day that
         are produced.
        """
        if lateral_length_ft is None:
            lateral_length_ft = self.lateral_length_ft
        if lateral_length_ft is None:
            raise ValueError('Must specify lateral length.')
        return np.exp(self.a + self.b * cumulative_elapsed_days) * lateral_length_ft

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
