import pandas as pd
import numpy as np

from utils import get_prod_window

__all__ = [
    'ExpRegressionModel',
    'dataframe_to_models',
]


class ExpRegressionModel:
    """
    A weighted exponential regression model for monthly production
    records (in BBLs/calendar day), adjusted for length of the lateral
    leg of the well.
    """

    # Results if 0.00000001 BBLs produced (before adjusting for lateral).
    A_NO_PROD = -18.42068074395236
    B_NO_PROD = -6.079362394156076e-16

    def __init__(
            self,
            a: float = np.nan,
            b: float = np.nan,
            lateral_length_ft: float = None,
            sufficient_data: bool = None,
            has_produced: bool = None,
            min_months: int = 36,
            max_months: int = 48,
            discard_gaps: bool = True,
            actual_months: int = None,
            weight_function: callable = None,
    ):
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
        :param sufficient_data: Whether the production records contained
         enough data for training (default min threshold is 36 months).
        :param has_produced: Whether the well in question has actually
         produced.
        :param min_months: Minimum number of months of production to
         require for training. Defaults to 36. (Note: If
          ``discard_gaps`` is used, will require this many months of
          actual production.)
        :param max_months: Maximum number of months of production to
         consider for training.
        :param actual_months: The number of months that were actually
         used for training. (Only use this parameter when loading a
         pre-trained model.)
        """
        self.a = a
        self.b = b
        self.min_months = min_months
        self.max_months = max_months
        self.discard_gaps = discard_gaps
        self.lateral_length_ft = lateral_length_ft
        self.has_produced = has_produced
        self.sufficient_data = sufficient_data
        self.actual_months = actual_months
        self.weight_function = weight_function

    def train(
            self,
            prod_records: pd.DataFrame,
            lateral_length_ft: float = None,
            min_months: int = 36,
            max_months: int = 48,
            discard_gaps: bool = True,
            weight_function: callable = None,
    ) -> (float, float):
        """
        Run a weighted exponential regression on monthly production
        records. Will use at least ``min_months`` and up to
        ``max_months``, if available.  If sufficient months of
        production are found in the provided records, will store
        ``.sufficient_records`` attribute as ``True``. Otherwise, will
        store ``False``.

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
        :param min_months: The fewest number of months of production to
         accept. If there are not enough, this will return ``None``.
        :param max_months: The max number of months of production to
         include.
        :param discard_gaps: Whether to discard or keep months during
         which no production occurred. Defaults to ``True`` (i.e.,
         discard).
        :param weight_function: (Optional) A function to apply to the
         monthly production values to determine their weighting in the
         exponential regression.
        :return: The ``a`` and ``b`` values to be used in the
         ``.predict...()`` method, which are also stored as attributes.
        """
        if lateral_length_ft is None:
            lateral_length_ft = self.lateral_length_ft
        else:
            self.lateral_length_ft = lateral_length_ft
        if lateral_length_ft is None:
            raise ValueError('Must specify lateral length.')
        self.has_produced = sum(prod_records['Oil Produced']) > 0
        self.sufficient_data = True
        selected = get_prod_window(prod_records, min_months, max_months, discard_gaps)
        if selected is None:
            # Not enough data.
            self.sufficient_data = False
            return None

        # Swap 0's for arbitrarily small values to avoid log issues.
        selected['bbls_per_prod_day'] = selected['bbls_per_prod_day'].replace(0, 0.0000001)
        selected['bbls_per_calendar_day'] = selected['bbls_per_calendar_day'].replace(0, 0.0000001)
        x = selected['calendar_days'].cumsum()
        y = selected['bbls_per_calendar_day'][x.index.values] / lateral_length_ft
        self.actual_months = len(x)

        weights = None
        if weight_function is None:
            weight_function = self.weight_function
        if weight_function is not None:
            weights = weight_function(y)
        poly_coefs = np.polynomial.Polynomial.fit(x, np.log(y), 1, w=weights)
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
        if round(sum(weights), 8) != 1.0:
            raise ValueError('Sum of `weights` must equal 1.0')
        a = 0.0
        b = 0.0
        for model, weight in zip(models, weights):
            a += model.a * weight
            b += model.b * weight
        return ExpRegressionModel(a=a, b=b)

    @staticmethod
    def daily_bbls_to_monthly(bbls_per_day: list[float], days_per_month: list[int]) -> list[float]:
        """
        Convert BBLs/calendar day to BBLs/month.

        Note: Both parameters must be equal in length.

        :param bbls_per_day: List of (predicted or actual) BBLs/calendar
         day.
        :param days_per_month: List of calendar days in each month.
        :return:
        """
        if len(bbls_per_day) != len(days_per_month):
            raise IndexError("Length of `bbls_per_day` and `days_per_month` must be equal.")
        return [bbls * days for bbls, days in zip(bbls_per_day, days_per_month)]


def dataframe_to_models(df: pd.DataFrame, api_nums: list[str] = None) -> dict[str, ExpRegressionModel]:
    """
    Load models from a dataframe of already-trained models. Optionally
    limit the results to the desired wells by passing a list of API
    numbers as ``api_nums``.
    :param df: A dataframe of already-trained models.
    :param api_nums: (Optional) A list of API numbers, represented as
     strings.
    :return: A dict of ``{API number: ExpRegressionModel}``.
    """
    if api_nums is None:
        api_nums = df['API_Label']
    models = {}
    for api_num in api_nums:
        row = df[df['API_Label'] == api_num]
        a = row['a'].values[0]
        b = row['b'].values[0]
        has_produced = row['has_produced'].values[0]
        lat_len = row['lateral_length_ft'].values[0]
        model = ExpRegressionModel(a=a, b=b, has_produced=has_produced, lateral_length_ft=lat_len)
        models[api_num] = model
    return models
