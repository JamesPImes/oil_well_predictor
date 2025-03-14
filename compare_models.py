import os
from pathlib import Path
from datetime import datetime

import dotenv
import pandas as pd
import numpy as np

from utils import (
    json_to_nearest_wells_params,
    json_to_exp_regress_params,
    get_prod_window,
)
from preprocess.prod_records.monthly_prod import ProductionPreprocessor
from model.exp_regress.exp_regress import ExpRegressionModel, dataframe_to_models
from model.nearest_wells.nearest_wells import (
    find_k_nearest,
    idw_weighting,
)

dotenv.load_dotenv()


class ModelComparer:
    well_data_fp = Path(os.getenv('WELL_DATA_FP'))
    model_dir = Path(os.getenv('MODEL_DIR'))
    comparison_results_dir = Path(os.getenv('COMPARISON_RESULTS_DIR'))
    production_records_dir = Path(os.getenv('PRODUCTION_RECORDS_DIR'))
    production_csv_template = "{api_num}_production_data.csv"
    training_test_data_dir = comparison_results_dir / 'testing_and_training_data'

    def __init__(self):
        self.comparison_results_dir.mkdir(exist_ok=True)
        self.training_test_data_dir.mkdir(exist_ok=True)
        self.existing_comparisons = [
            fn for fn in os.listdir(self.comparison_results_dir) if fn.endswith('.csv')
        ]
        wells = pd.read_csv(self.well_data_fp, parse_dates=['Spud_Date', 'Stat_Date'])
        self.wells = wells.dropna(subset=['lateral_length_ft'])
        # Data is split into training / test data.
        self.test_wells = None
        self.training_wells = None
        wells.to_csv(self.training_test_data_dir / 'all_wells.csv', index=True)

        # Load pre-trained exponential regression models for each well.
        self.expreg_models_all = {}
        self.load_exp_regress_models()

        self.near_wells_params = json_to_nearest_wells_params("nearest_wells_parameters.json")
        self.exp_reg_params = json_to_exp_regress_params("exp_regress_model_parameters.json")

        # Store results in a dict, keyed by tuple: (near_wells_model_name, expreg_model_name).
        self.results = {}
        self.results_dfs = {}
        for near_well_model_name in self.near_wells_params.keys():
            for expreg_model_name in self.expreg_models_all.keys():
                self.results[(near_well_model_name, expreg_model_name)] = []
                self.results_dfs[(near_well_model_name, expreg_model_name)] = None

    def split_training_and_test_data(self, training_frac=0.2, random_state=None) -> None:
        """
        Split the dataset into testing and training data. Save the split
        to disk.
        """
        self.test_wells = self.wells.sample(frac=training_frac, random_state=random_state)
        self.training_wells = self.wells.drop(self.test_wells.index)
        self.test_wells.to_csv(self.training_test_data_dir / 'test_wells.csv', index=True)
        self.training_wells.to_csv(self.training_test_data_dir / 'training_wells.csv', index=True)

    def load_exp_regress_models(self) -> dict:
        """Load already-trained exponential regression models."""
        for model_fn in os.listdir(self.model_dir):
            if not model_fn.lower().endswith('.csv'):
                continue
            model_fp = self.model_dir / model_fn
            model_name = model_fp.stem
            model_df = pd.read_csv(model_fp)
            self.expreg_models_all[model_name] = dataframe_to_models(model_df)
        return self.expreg_models_all

    def load_prior_testing_training_data(self):
        self.test_wells = pd.read_csv(self.training_test_data_dir / 'test_wells.csv', index_col=0)
        self.training_wells = pd.read_csv(self.training_test_data_dir / 'training_wells.csv', index_col=0)
        self.wells = pd.read_csv(self.training_test_data_dir / 'all_wells.csv', index_col=0)

    def load_prior_results(self):
        """Load prior results (CSV files) into the ``.results`` dict."""
        for fn in os.listdir(self.comparison_results_dir):
            if not fn.lower().endswith('.csv'):
                continue
            model_pair_raw = fn[:-4].split('__')
            near_well_model_name, expreg_model_name = model_pair_raw
            results_df = pd.read_csv(self.comparison_results_dir / fn)
            self.results_dfs[(near_well_model_name, expreg_model_name)] = results_df
        self.load_prior_testing_training_data()

    def results_to_dfs(self) -> dict:
        for model_pair, result in self.results.items():
            self.results_dfs[model_pair] = pd.DataFrame(data=result)
        return self.results_dfs

    def run_all_models(self):
        for near_well_model_name, param_set in self.near_wells_params.items():
            k = param_set['k']
            i = 0
            n = len(self.test_wells)
            for _, row in self.test_wells.iterrows():
                row_timestamp = datetime.now()
                i += 1
                api_num = row['API_Label']
                lat_len = row['lateral_length_ft']
                print(f"{near_well_model_name} [{i}/{n}] -- {api_num}", end=" ")
                nearest = find_k_nearest(
                    self.training_wells,
                    target_shl=(row['lat_shl'], row['long_shl']),
                    target_bhl=(row['lat_bhl'], row['long_bhl']),
                    # k+1 because we expect to capture the well itself, if
                    # it was left in the training data.
                    k=k + 1
                )
                try:
                    # Pop the target well itself, if found.
                    popped = nearest.pop(api_num)
                except KeyError:
                    # Pop the smallest instead, to get back down to k elements.
                    nearest.pop(min(nearest, key=nearest.get))

                actual_prod_raw = pd.read_csv(
                    self.production_records_dir / self.production_csv_template.format(api_num=api_num),
                    parse_dates=['First of Month'],
                )
                prepro = ProductionPreprocessor(actual_prod_raw)
                actual_prod = prepro.preprocess_all()

                for expreg_model_name, expreg_model in self.expreg_models_all.items():
                    if f"{near_well_model_name}__{expreg_model_name}.csv" in self.existing_comparisons:
                        continue
                    print(",", expreg_model_name, end="")
                    # Use the original params to ensure we compare apples-to-apples
                    # (i.e., use the same actual production window as was used to train).
                    params = self.exp_reg_params[expreg_model_name]
                    selected_prod_records = get_prod_window(
                        actual_prod,
                        min_months=params['min_months'],
                        max_months=params['max_months'],
                        discard_gaps=params['discard_gaps']
                    )
                    if selected_prod_records is None:
                        # Not enough data.
                        result = {
                            'API_Label': api_num,
                            'months': None,
                            'predicted': None,
                            'actual': None,
                            'abs_error': None,
                            'rel_error': None,
                            'predicted/actual': None,
                            'prediction_time_cost_sec': (row_timestamp - datetime.now()).microseconds / 1_000_000,
                            'component_models': None,
                            'a': None,
                            'b': None,
                            'mean_dist': None,
                            'median_dist': None,
                            'std_dist': None,
                            # 'predicted2': pred_total_by_individual,
                            # 'abs_error2': abs(actual_total - pred_total_by_individual),
                            # 'rel_error2': abs(actual_total - pred_total_by_individual) / actual_total,
                            'comment': "insufficient data"
                        }
                        self.results[(near_well_model_name, expreg_model_name)].append(result)
                        continue

                    monthly_days = selected_prod_records['calendar_days']
                    # Grab the production models from the k-nearest wells and compute
                    # the weighting for each to create a weighted model for our target
                    # well.
                    nearby_expreg_models = [expreg_model[nearby_api_num] for nearby_api_num in nearest.keys()]
                    weights = idw_weighting(distances=nearest, power=param_set['distance_weighting'])
                    component_models = [
                        f"{api}<dist_{nearest[api]};wt_{weights[api]};a_{mdl.a};b_{mdl.b}>"
                        for api, mdl in zip(nearest.keys(), nearby_expreg_models)
                    ]
                    distances = list(nearest.values())
                    target_model = ExpRegressionModel.weight_models(nearby_expreg_models, weights.values())
                    pred_cumulative_days = pd.Series(range(1, sum(monthly_days) + 1))
                    predicted_bbls_each_day = target_model.predict_bbls_per_calendar_day(
                        pred_cumulative_days, lateral_length_ft=lat_len)
                    predicted_total = sum(predicted_bbls_each_day)
                    actual_total = sum(selected_prod_records['Oil Produced'])

                    # # For additional prediction and comparison, calculate each component
                    # # exp. regression model's own prediction for this timeframe.
                    # pred_total_by_individual = 0
                    # for m, wt in zip(nearby_expreg_models, weights.values()):
                    #     pred = m.predict_bbls_per_calendar_day(pred_cumulative_days, lateral_length_ft=lat_len)
                    #     pred_total_by_individual += sum(pred) * wt

                    result = {
                        'API_Label': api_num,
                        'months': len(selected_prod_records),
                        'predicted': predicted_total,
                        'actual': actual_total,
                        'abs_error': abs(actual_total - predicted_total),
                        'rel_error': abs(actual_total - predicted_total) / actual_total,
                        'predicted/actual': predicted_total / actual_total,
                        'prediction_time_cost_sec': (row_timestamp - datetime.now()).microseconds / 1_000_000,
                        'component_models': '|'.join(component_models),
                        'a': target_model.a,
                        'b': target_model.b,
                        'mean_dist': np.mean(distances),
                        'median_dist': np.median(distances),
                        'std_dist': np.std(distances),
                        # 'predicted2': pred_total_by_individual,
                        # 'abs_error2': abs(actual_total - pred_total_by_individual),
                        # 'rel_error2': abs(actual_total - pred_total_by_individual) / actual_total,
                        'comment': None,
                    }
                    self.results[(near_well_model_name, expreg_model_name)].append(result)
                print("")

            # Output each model's results to its own table.
            for model_pair, result in self.results.items():
                if model_pair[0] != near_well_model_name:
                    # Write as we go.
                    continue
                fn = f"{model_pair[0]}__{model_pair[1]}.csv"
                results_df = pd.DataFrame(data=result)
                results_df.to_csv(self.comparison_results_dir / fn, index=False)

        self.results_to_dfs()
        self.gen_comparison_report()
        return None

    @staticmethod
    def parse_model_components_str(s) -> list[dict]:
        """
        Parse the stored model component string into a list of dicts of
        those model components.
        """
        # api_num<dist_FLOAT;wt_FLOAT;a_FLOAT;b_FLOAT>|...
        model_components = []
        models_raw = s.split('|')
        for model_raw in models_raw:
            # Strip brackets.
            api_num, etc = model_raw[:-1].split('<')
            dist_raw, wt_raw, a_raw, b_raw = etc.split(';')
            model = {
                'api_num': api_num,
                'dist': float(dist_raw[5:]),
                'wt': float(wt_raw[3:]),
                'a': float(a_raw[2:]),
                'b': float(b_raw[2:])
            }
            model_components.append(model)
        return model_components

    def gen_comparison_report(self):
        comparisons = []
        for model_pair, results_df in self.results_dfs.items():
            near_well_model_name, expreg_model_name = model_pair
            param_set = self.near_wells_params[near_well_model_name]
            meaningful_results = results_df.dropna(subset=['predicted'])
            mean_error = (meaningful_results['predicted'] - meaningful_results['actual']).mean()
            mean_actual_bbls = meaningful_results['actual'].mean()
            comparison_entry = {
                'near_well_model_name': near_well_model_name,
                'expreg_model_name': expreg_model_name,
                'k': param_set['k'],
                'distance_weighting': param_set['distance_weighting'],
                'avg_sec_per_predict': meaningful_results['prediction_time_cost_sec'].mean(),
                'mean_abs_error': meaningful_results['abs_error'].mean(),
                'mean_error': mean_error,
                'mean_abs_pct_error': meaningful_results['rel_error'].mean(),
                'mean_pct_error': (meaningful_results['predicted/actual'] - 1).mean(),
                'total_actual_bbls': meaningful_results['actual'].sum(),
                'total_predicted_bbls': meaningful_results['predicted'].sum(),
                'mean_actual_bbls': mean_actual_bbls,
                'mean_predicted_bbls': meaningful_results['predicted'].mean(),
                'mean_error/mean_actual_bbls': mean_error / mean_actual_bbls,
                # 'mean_abs_error2': meaningful_results['abs_error2'].mean(),
                # 'mean_rel_error2': meaningful_results['rel_error2'].mean(),
                'total_predicted': len(meaningful_results),
            }
            comparisons.append(comparison_entry)
        comparison_df = pd.DataFrame(data=comparisons)
        comparison_df.to_csv('comparison.csv', index=False)

    def fill_distance_calcs(self):
        for fn in os.listdir(self.comparison_results_dir):
            fp = self.comparison_results_dir / fn
            if fp.suffix != '.csv':
                continue
            print('Processing...', fn)
            df = pd.read_csv(fp)
            mean_dists = []
            median_dists = []
            std_dists = []
            for comp_model_str in df['component_models']:
                comp_models = self.parse_model_components_str(comp_model_str)
                dists = [md['dist'] for md in comp_models]
                mean_dists.append(np.mean(dists))
                median_dists.append(np.median(dists))
                std_dists.append(np.std(dists))
            i = len(df.columns) - 1
            df.insert(i, 'std_dist', pd.Series(std_dists))
            df.insert(i, 'median_dist', pd.Series(median_dists))
            df.insert(i, 'mean_dist', pd.Series(mean_dists))
            df.to_csv(fp, index=False)


if __name__ == '__main__':
    ...
