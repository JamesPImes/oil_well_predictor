
import pandas as pd

from .exp_regress import ExpRegressionModel
from .nearest_wells import (
    KNearestWellsFinder,
    DistanceCalculator,
    get_distance_calculator,
    idw_weighting,
)

class CompositeModelBuilder:
    """
    Perform KNN and build a composite exponential regression model from
    the results.

    Note: Create a separate builder for each combination of ``k``,
    ``training_wells``, and the component exponential regression models
    (as trained on its own parameter set).
    """

    def __init__(
            self,
            training_wells: pd.DataFrame,
            k: int,
            all_exp_regress_model_coefs: list[dict],
            distance_calculator: DistanceCalculator = None,
            all_wells: pd.DataFrame = None,
    ):
        """
        :param training_wells: A dataframe of the wells in the training
         data. (Should include ``'API_Label'`` and the usual location
         data.)

        :param k: How many nearby neighbors to include.

        :param all_exp_regress_model_coefs: A dict keyed by API number,
         whose values are a nested dict of ``{'a': <float>, 'b': <float>}``
         coefficients for the trained exponential regression models.
         Should contain a model for at least every well in the training
         set. (Use the ``CompositeModelBuilder.extract_coefs_from_df()``
         method to get this from a dataframe of models.)

        :param distance_calculator: A ``DistanceCalculator`` instance.
         If not pass, will create one for the ``training_wells``.
         (Create one prior to initializing this instance in order to
         take advantage of caching.)

        :param all_wells: A dataframe of well data for all available
         wells (whether in the training dataset or not). Necessary to
         use the ``knn_well()`` method, in order to pull its location.
        """
        self.training_wells = training_wells
        self.k = k
        if all_wells is None:
            all_wells = training_wells
        self.all_wells = all_wells
        self.all_exp_regress_model_coefs = all_exp_regress_model_coefs
        if distance_calculator is None:
            distance_calculator = get_distance_calculator(all_wells)
        self.distance_calculator = distance_calculator
        self.knn_finder = KNearestWellsFinder(training_wells, distance_calculator)

    def knn_location(self, location: tuple[float, float], idw_power=None) -> ExpRegressionModel:
        """
        Create a composite exponential regression model from the
        k-nearest wells in the training data (calculated from a
        particular lat/long coord).

        :param location: Lat/long coord (tuple) of a target.
        :param idw_power: (Optional) The power to use in the IDW
         function. Higher power results in stronger weighting toward
         nearby wells. If not specified (None or 0), will return equal
         weights.
        :return: An ``ExpRegressionModel`` with weighted coefficients
         from the models for the k-nearest wells. (Still needs user to
         specify the lateral length when making predictions.)
        """
        # api_number :: distance
        wells_distances = self.knn_finder.find_nearest_from_location(location, self.k)
        if idw_power is None:
            idw_power = 0
        # api_number :: weighting
        wells_weights = idw_weighting(wells_distances, idw_power)
        # [{'a': -3.8632, 'b': -0.00165}, {'a': -2.9075, 'b': -0.00244}, ...]
        models = []
        # [0.28532, 0.18392, ...]
        weights = []
        for api_number, weight in wells_weights.items():
            models.append(self.all_exp_regress_model_coefs[api_number])
            weights.append(weight)
        final_model = ExpRegressionModel.composite_model(models, weights)
        return final_model

    def knn_well(self, api_num: str, idw_power=None) -> ExpRegressionModel:
        """
        Create a composite exponential regression model from the
        k-nearest wells in the training data (calculated from a
        particular well).

        WARNING: To use this method, the API number must be in the
        distance calculator passed at init. (If the distance calculator
        was not passed, this would still work if ``all_wells`` was
        passed at init.)

        :param api_num: API number of the target well.
        :param idw_power: (Optional) The power to use in the IDW
         function. Higher power results in stronger weighting toward
         nearby wells. If not specified (None or 0), will return equal
         weights.
        :return: An ``ExpRegressionModel`` with weighted coefficients
         from the models for the k-nearest wells. (Still needs user to
         specify the lateral length when making predictions.)
        """
        location = self.distance_calculator.locations[api_num]
        return self.knn_location(location, idw_power=idw_power)

    @staticmethod
    def extract_coefs_from_df(
            models: pd.DataFrame,
            api_header='API_Label'
    ) -> dict:
        """
        Helper function to extract the ``a`` and ``b`` coefficients for
        exponential regression models from a dataframe.

        :param models: A dataframe with API numbers, and ``'a'`` and
         ``'b'`` values from previously trained exponential regression
         models.
        :param api_header: The header for the column containing API
         numbers.
        :return: A dict, keyed by API number, whose values are a nested
         dict of ``{'a': <float>, 'b': <float>}`` coefficients.
        """
        all_model_coefs = {}
        for _, row in models.iterrows():
            api_num = row[api_header]
            all_model_coefs[api_num] = {'a': row['a'], 'b': row['b']}
        return all_model_coefs
