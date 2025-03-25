import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from model import CompositeModelBuilder
from utils import get_cumulative_production, get_cumulative_days
from preprocess.prod_records import ProductionLoader

__all__ = [
    'run_predictions_at_checkpoints',
    'evaluate',
    'kfold_cv',
]


def run_predictions_at_checkpoints(
    train_wells: pd.DataFrame,
    test_wells: pd.DataFrame,
    production_loader: ProductionLoader,
    exp_reg_models: pd.DataFrame,
    knn_k: int,
    idw_power: float,
    distance_calculator=None,
    month_checkpoints: list = (24, 36, 48),
) -> pd.DataFrame:
    """
    Run the predictions at the specified checkpoints (default is 24, 36,
    and 48 months). Get back a dataframe containing the API numbers of
    the wells in the test data, and columns for the actual and predicted
    total production at each checkpoint. Those column headers have the
    format ``'true_##'`` and  ''`pred_##`'' (where ``'##'`` would be
    replaced with the number of months at that checkpoint -- e.g.,
    ``'pred_48'``).

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
    :param month_checkpoints: The months at which to calculate metrics.
     (Defaults to first 24, 36, and 48 months.)
    :return: A dataframe as described above.
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
            actual[chk].append(cumul_actual[chk])
            predicted[chk].append(cumul_pred[chk])
    output_df = test_wells[['API_Label']].copy()
    for chk in month_checkpoints:
        output_df[f"true_{chk}"] = actual[chk]
        output_df[f"pred_{chk}"] = predicted[chk]
    return output_df


def evaluate(
    predictions: pd.DataFrame,
    month_checkpoints: list = (24, 36, 48),
    metrics: dict = None,
):
    """
    Calculate scores for the requested metrics using the test data.
    (To calculate scores on the training, pass the same set of wells
    as ``train_wells`` and ``test_wells``.)

    :param predictions: A dataframe containing results from
     ``run_predictions_at_checkpoints()``.
    :param month_checkpoints: A list of month checkpoints (defaults to
     24, 36, and 48).
    :param metrics: A dict, whose keys are metric names and whose values
     are functions to calculate them (should take actual total
     production and predicted production).
    :return: A dataframe with the scores on each metric at each
     checkpoint. Fields are ``'month_checkpoint'`` and
     ``'<metric_name>_##'`` (where ``##`` corresponds to each
     checkpoint) for each metric/checkpoint pair.

     Note that if there was insufficient data for a given well at a
     requested checkpoint (e.g., a well produced only 47 months, but 48
     months was requested), then that well will be dropped from metric
     calculations. If no wells produced the requested number of months,
     then all scores for that checkpoint will be ``None``.
    """
    score_data = {'month_checkpoint': []}
    for metric_name in metrics.keys():
        score_data[metric_name.upper()] = []
    for checkpoint in month_checkpoints:
        score_data['month_checkpoint'].append(checkpoint)
        chk_actual = np.array(predictions[f"true_{checkpoint}"])
        chk_actual = chk_actual[~np.isnan(chk_actual)]
        chk_pred = np.array(predictions[f"pred_{checkpoint}"])
        chk_pred = chk_pred[~np.isnan(chk_pred)]
        for metric_name, func in metrics.items():
            # In case none of the wells had sufficient data at this checkpoint...
            res = None
            if len(chk_actual) > 0:
                res = func(chk_actual, chk_pred)
            score_data[metric_name.upper()].append(res)
    return pd.DataFrame(score_data)


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
        fold_results = run_predictions_at_checkpoints(
            train_wells,
            test_wells,
            production_loader=production_loader,
            exp_reg_models=exp_reg_models,
            knn_k=knn_k,
            idw_power=idw_power,
            distance_calculator=distance_calculator,
            month_checkpoints=month_checkpoints,
        )
        fold_scores = evaluate(fold_results, month_checkpoints, metrics)
        for chkpt in month_checkpoints:
            chk_scores = fold_scores.loc[fold_scores['month_checkpoint'] == chkpt].iloc[0]
            subresults_dict = checkpoint_results[chkpt]
            for metric_name in metrics.keys():
                score = chk_scores[metric_name]
                if score is None or np.isnan(score):
                    continue
                subresults_dict[metric_name].append(score)
    # Find averages for each checkpoint/metric.
    for chkpt, chkpt_metrics in checkpoint_results.items():
        for metric_name, scores in chkpt_metrics.items():
            metric_avgs[chkpt][metric_name] = np.mean(scores)
    return metric_avgs
