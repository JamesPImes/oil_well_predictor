
"""
Some functions used in the demo notebook. Put here to keep the notebook
cleaner.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from model.exp_regress import train_all_exp_regress
from model.nearest_wells import DistanceCalculator
from preprocess.prod_records import ProductionLoader
import evaluate

__all__ = [
    'convert_tuning_results_to_df',
    'rank_best',
    'metrics_heatmaps',
    'prepare_exp_regress_models',
    'grid_search',
]


def convert_tuning_results_to_df(
        model_scores: dict,
        metric_names: list = ('MAPE', 'MAE', 'RMSE'),
        month_checkpoints: list = (24, 36, 48)
) -> pd.DataFrame:
    """Convert the CV results to a dataframe."""
    metric_names = [mn.lower() for mn in metric_names]
    tuning_results = {
        'param_set': [],
        'exp_reg_weight': [],
        'knn_k': [],
        'idw_power': []
    }
    for metric_name in metric_names:
        for chkpt in month_checkpoints:
            tuning_results[f"{metric_name}_{chkpt}"] = []
    for param_set, score_dicts in model_scores.items():
        exp_reg_wt, knn_k, idw_power = param_set
        tuning_results['param_set'].append(param_set)
        tuning_results['exp_reg_weight'].append(exp_reg_wt)
        tuning_results['knn_k'].append(knn_k)
        tuning_results['idw_power'].append(idw_power)
        for metric_name in metric_names:
            for chkpt in month_checkpoints:
                score = score_dicts[chkpt][metric_name]
                tuning_results[f"{metric_name}_{chkpt}"].append(float(score))
    return pd.DataFrame(tuning_results)

def rank_best(results_df: pd.DataFrame, metric_name: str, month_checkpoint: int, ideal='MIN'):
    """Rank the parameter sets by their score for the chosen metric."""
    target = f"{metric_name}_{month_checkpoint}"
    cols = ['param_set', 'exp_reg_weight', 'knn_k', 'idw_power', target]
    relevant_scores = results_df[cols]

    # Assume min.
    ascending = True
    if ideal == 'MAX':
        ascending = False
    relevant_scores = relevant_scores.sort_values(
        by=target, ascending=ascending, ignore_index=True)
    return relevant_scores

def metrics_heatmaps(
    tuning_results_subset: pd.DataFrame,
    metric_names: list = ('MAPE', 'MAE', 'RMSE'),
    month_checkpoints: list = (24, 36, 48),
    cmap: str = 'mako',
    heatmap_txt_fmt: dict = None
):
    """
    :param tuning_results_subset: A subset of the tuning_results
     dataframe, with a single ``'exp_regress_weight'`` value, and
     limited to whichever ``'knn_k'`` and ``'idw_power'`` values are
     desired.
    :param metric_names: Which metrics to include.
    :param month_checkpoints: Which months to include.
    :param cmap: The color map to use. (Defaults to ``'mako'``)
    :param heatmap_txt_fmt: A dict of {<metric_name>: <format string>}
     to pass to ``sns.heatmap(..., fmt=<here>)`` for each metric.
    """
    checkpoints = month_checkpoints

    fig, axs = plt.subplots(len(month_checkpoints), len(metric_names), figsize=(16, 16))
    if heatmap_txt_fmt is None:
        # Some defaults.
        heatmap_txt_fmt = {'MAPE': '.4f', 'MAE': '.0f', 'RMSE': '.0f'}
    row = 0
    for metric_name in metric_names:
        col = 0
        for chkpt in checkpoints:
            ax = axs[row][col]
            col_header = f"{metric_name}_{chkpt}".lower()
            piv = tuning_results_subset.pivot_table(
                values=col_header, index=['knn_k'], columns=['idw_power'])
            sns.heatmap(
                piv,
                ax=ax,
                annot=True,
                fmt=heatmap_txt_fmt.get(metric_name),
                cmap=cmap,
                annot_kws={"size": 8}
            )
            ax.set_title(f"{metric_name} ({chkpt} months)", fontsize=12)
            col += 1
        row += 1
    plt.show()
    return None

def prepare_exp_regress_models(
    wells_all: pd.DataFrame,
    production_records_dir: Path,
    exp_regress_dir: Path,
    exp_reg_weight_power_vals: list = (2.0, 1.5, 1.0, 0.75, 0.5, 0.375, 0.25, 0.125),
    production_csv_fn_template: str = "{api_num}_production_data.csv",
    exp_regress_fn_template: str = "exp_regress_models_{weight}.csv",
):
    """
    Prepare all the exponential regression models for every configured
    weight power.

    :param wells_all: A fully preprocessed dataframe of wells (including
     all training and testing wells).
    :param production_records_dir: The path to the directory containing
     production records.
    :param exp_regress_dir: The filepath where trained models will be
     saved to and loaded from.
    :param exp_reg_weight_power_vals: A list of floats corresponding to
     the weight powers for the exponential regression models.
    :param production_csv_fn_template: The template for production
     record filenames.  Default: ``"{api_num}_production_data.csv"``
    :param exp_regress_fn_template: The template for the filename for
     exponential regression models.
     Default: ``"exp_regress_models_{weight}.csv"``
    :return: A dict mapping weights to dataframes of the model
     coefficients.
    """
    exp_regress_dir = Path(exp_regress_dir)
    print("Training initial exponential regression models...")
    exp_reg_models_by_weights = {}
    for wt in exp_reg_weight_power_vals:
        save_fp = exp_regress_dir / exp_regress_fn_template.format(weight=wt)
        if save_fp.exists():
            print(f" -- Loading existing models with weight {wt}...")
            models = pd.read_csv(save_fp)
        else:
            print(f" -- Training all with weight {wt}...")
            models = train_all_exp_regress(
                wells_all,  # all wells (train + test)
                prod_records_dir=production_records_dir,
                prod_csv_template=production_csv_fn_template,
                output_fp=save_fp,  # Save results to csv.
                weight_power=wt,
            )
        exp_reg_models_by_weights[wt] = models
    print("Initial exponential regression models complete.\n")
    return exp_reg_models_by_weights

def grid_search(
    wells_train: pd.DataFrame,
    tuning_results_fp: Path,
    production_loader: ProductionLoader,
    distance_calculator: DistanceCalculator,
    metrics: dict,
    exp_reg_models_by_weights: dict,
    exp_reg_weight_power_vals: list = (2.0, 1.5, 1.0, 0.75, 0.5, 0.375, 0.25, 0.125),
    knn_k_vals: list = (2, 3, 4, 5, 6, 7, 8),
    idw_power_vals: list = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0),
    n_splits: int = 10,
    month_checkpoints: list = (24, 36, 48),
    random_state: int = 42,
):
    """
    Perform grid search to find optimal hyperparameters

    :param wells_train: A wells dataframe, limited to wells in the
     training data.
    :param tuning_results_fp: A filepath where tuning results will be
     saved.
    :param production_loader: A production loader.
    :param distance_calculator: A distance calculator.
    :param metrics: A dict of ``{<metric_name>: <function>}``.
    :param exp_reg_models_by_weights: A dict of
     ``{<exp_reg_weight>: <dataframe of model coefficients>}``
    :param exp_reg_weight_power_vals: A list of floats corresponding to
     the weight powers for the exponential regression models. (The
     ``exp_reg_models_by_weights`` dict should have keys for all of
     these values.)
    :param knn_k_vals: A list of ``k`` values for KNN.
    :param idw_power_vals: A list of floats corresponding to the power
     to use in IDW for KNN.
    :param n_splits: Number of folds to use for training. Default: 10.
    :param month_checkpoints: A list of month checkpoints.
     Default: [24, 36, 48]
    :param random_state:
    :return: A dataframe containing all of the parameter sets and their
     scores.
    """
    tuning_results_fp = Path(tuning_results_fp)
    # Load existing results, if any.
    existing_tuning_results = pd.DataFrame({'param_set': []})
    prior_param_sets = []
    if tuning_results_fp.exists():
        print("Loading existing tuning results...\n")
        existing_tuning_results = pd.read_csv(tuning_results_fp)
        etr = existing_tuning_results
        # Recreate param_set tuple.
        etr['param_set'] = list(zip(etr['exp_reg_weight'], etr['knn_k'], etr['idw_power']))
        prior_param_sets = etr['param_set'].to_list()

    print("Beginning cross-validation...")
    model_scores = {}
    for exp_reg_weight in exp_reg_weight_power_vals:
        expreg_models = exp_reg_models_by_weights[exp_reg_weight]
        for knn_k in knn_k_vals:
            for idw_power in idw_power_vals:
                param_set = (exp_reg_weight, knn_k, idw_power)
                if param_set in prior_param_sets:
                    # Avoid recalculating existing parameter sets.
                    continue
                print(f"-- exp.reg: {exp_reg_weight}, k: {knn_k}, idw: {idw_power}")
                results = evaluate.kfold_cv(
                    wells=wells_train,  # only training wells for tuning.
                    production_loader=production_loader,
                    n_splits=n_splits,
                    exp_reg_models=expreg_models,
                    knn_k=knn_k,
                    idw_power=idw_power,
                    distance_calculator=distance_calculator,
                    month_checkpoints=month_checkpoints,
                    metrics=metrics,
                    random_state=random_state,
                )
                model_scores[param_set] = results
    print("All cross-validation complete.")

    if model_scores:
        tuning_results = convert_tuning_results_to_df(model_scores)
        if len(existing_tuning_results) > 0:
            tuning_results = pd.concat([existing_tuning_results, tuning_results], ignore_index=True)
        # Save ourselves an hour of compute by saving the results.
        tuning_results.to_csv(tuning_results_fp, index=False)
    else:
        tuning_results = existing_tuning_results
    return tuning_results
