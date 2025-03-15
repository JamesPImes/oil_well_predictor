import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from model import CompositeModelBuilder
from utils import get_cumulative_production, get_cumulative_days
from preprocess.prod_records import ProductionLoader

__all__ = [
    'evaluate',
    'kfold_cv',
]


def evaluate(
    train_wells: pd.DataFrame,
    test_wells: pd.DataFrame,
    production_loader: ProductionLoader,
    exp_reg_models: pd.DataFrame,
    knn_k: int,
    idw_power: float,
    distance_calculator=None,
    month_checkpoints: list = (24, 36, 48),
    metrics: dict = None,
):
    """
    Calculate scores for the requested metrics using the test data.
    (To calculate scores on the training, pass the same set of wells
    as ``train_wells`` and ``test_wells``.)

    :param train_wells: A dataframe of wells in training set.
    :param test_wells: A dataframe of wells in the test set.
    :param production_loader: A ``ProductionLoader`` object. (Should
     create the same production window as was used to train the model.)
    :param exp_reg_models: A dataframe of trained exponential regression
     models, with columns for the API number and the coefficients
     ``'a'`` and ``'b'``.
    :param knn_k: How many nearby neighbors to select.
    :param idw_power: How strongly to weight the KNN results by
     nearness.
    :param distance_calculator: A ``DistanceCalculator`` object, prepped
     with all relevant wells.
    :param metrics: A dict, whose keys are metric names and whose values
     are functions to calculate them (should take actual total
     production and predicted production).
    :param month_checkpoints: The months at which to calculate metrics.
     (Defaults to first 24, 36, and 48 months.)
    :return: A nested dict. Top-level keys are the month checkpoints.
     Internal dicts are the metric names and their scores.
     Ex:  ``{48: {'mape': 0.234, 'mae': 12345}, ...}``.

     Note that if there was insufficient data for a given well at a
     requested checkpoint (e.g., a well produced only 47 months, but 48
     months was requested), then that well will be dropped from metric
     calculations. If no wells produced the requested number of months,
     then all scores for that checkpoint be ``None``.
    """
    coefs = CompositeModelBuilder.extract_coefs_from_df(exp_reg_models)
    builder = CompositeModelBuilder(train_wells, knn_k, coefs, distance_calculator)

    # Keep a separate set of results for each checkpoint.
    actual = {checkpoint: [] for checkpoint in month_checkpoints}
    predicted = {checkpoint: [] for checkpoint in month_checkpoints}
    # Build a composite model for each test well, and extract the cumulative production
    # at each checkpoint.
    for _, row in test_wells.iterrows():
        target_api_num = row['API_Label']
        lat_len = row['lateral_length_ft']
        composite_model = builder.knn_well(target_api_num, idw_power=idw_power)
        actual_production = production_loader.load(target_api_num)
        cumulative_days = get_cumulative_days(actual_production)
        predicted_bbls = composite_model.predict_bbls_per_calendar_day(
            cumulative_days, lateral_length_ft=lat_len)
        cumul_actual, cumul_pred = get_cumulative_production(
            actual_production, predicted_bbls, month_checkpoints)
        for chk in month_checkpoints:
            if cumul_actual[chk] is None:
                # This well did not have the requested number of months of production.
                # Omit this well from our metrics for this checkpoint.
                continue
            actual[chk].append(cumul_actual[chk])
            predicted[chk].append(cumul_pred[chk])

    # Calculate each metric for each checkpoint.
    checkpoint_results = {chk: {} for chk in month_checkpoints}
    for checkpoint in month_checkpoints:
        chk_actual = actual[checkpoint]
        chk_pred = predicted[checkpoint]
        results_dict = checkpoint_results[checkpoint]
        for metric_name, func in metrics.items():
            # In case none of the wells had sufficient data at this checkpoint...
            res = None
            if chk_actual:
                res = func(chk_actual, chk_pred)
            results_dict[metric_name] = res
    return checkpoint_results


def kfold_cv(
    wells: pd.DataFrame,
    production_loader: ProductionLoader,
    exp_reg_models: pd.DataFrame,
    knn_k: int,
    idw_power: float,
    n_splits: int,
    distance_calculator=None,
    month_checkpoints: list = (24, 36, 48),
    metrics: dict = None,
    random_state=42,
):
    """
    Run K-fold cross-validation with the provided ``wells``.

    """
    # checkpoint_results --> {48: {'mape': [0.123, 0.234, ...], 'mae': [23123, 41232, ...] } }
    # metric_avgs --> {48: {'mape': 0.210, 'mae': 21345} }
    checkpoint_results = {}
    metric_avgs = {}
    # Build the skeleton of our main results and output dicts.
    for checkpoint in month_checkpoints:
        checkpoint_results[checkpoint] = {}
        metric_avgs[checkpoint] = {}
        for metric_name in metrics.keys():
            checkpoint_results[checkpoint][metric_name] = []
            metric_avgs[checkpoint][metric_name] = None
    # Run the cross-validation.
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for i, (train_idxs, test_idxs) in enumerate(kf.split(wells), start=1):
        train_wells = wells.iloc[train_idxs]
        test_wells = wells.iloc[test_idxs]
        fold_results = evaluate(
            train_wells,
            test_wells,
            production_loader=production_loader,
            exp_reg_models=exp_reg_models,
            knn_k=knn_k,
            idw_power=idw_power,
            distance_calculator=distance_calculator,
            metrics=metrics,
            month_checkpoints=month_checkpoints,
        )
        # Extract the results of this fold into the main results dict.
        for chkpt, chkpt_metrics in fold_results.items():
            subresults_dict = checkpoint_results[chkpt]
            for metric_name, score in chkpt_metrics.items():
                if score is None:
                    continue
                subresults_dict[metric_name].append(score)
    # Find averages for each checkpoint/metric.
    for chkpt, chkpt_metrics in checkpoint_results.items():
        for metric_name, scores in chkpt_metrics.items():
            metric_avgs[chkpt][metric_name] = np.mean(scores)
    return metric_avgs
