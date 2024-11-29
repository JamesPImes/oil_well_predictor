import os
from pathlib import Path
from datetime import datetime

import dotenv
import pandas as pd

from utils import (
    json_to_nearest_wells_params,
    json_to_exp_regress_params,
    get_prod_window,
)
from preprocess.prod_records.monthly_prod import ProductionPreprocessor
from model.prod_records.exp_regress import ExpRegressionModel, dataframe_to_models
from model.nearest_wells.nearest_wells import (
    find_k_nearest,
    idw_weighting,
)

dotenv.load_dotenv()
WELL_DATA_FP = Path(os.getenv('WELL_DATA_FP'))
MODEL_DIR = Path(os.getenv('MODEL_DIR'))
COMPARISON_RESULTS_DIR = Path(os.getenv('COMPARISON_RESULTS_DIR'))
COMPARISON_RESULTS_DIR.mkdir(exist_ok=True)
PRODUCTION_RECORDS_DIR = Path(os.getenv('PRODUCTION_RECORDS_DIR'))
PRODUCTION_CSV_TEMPLATE = "{api_num}_production_data.csv"

if __name__ == '__main__':
    start_time = datetime.now()
    existing_comparisons = [fn for fn in os.listdir(COMPARISON_RESULTS_DIR) if fn.endswith('.csv')]
    wells = pd.read_csv(WELL_DATA_FP, parse_dates=['Spud_Date', 'Stat_Date'])
    wells = wells.dropna(subset=['lateral_length_ft'])
    # 80/20 split between training / test data.
    test_wells = wells.sample(frac=0.2)
    training_wells = wells.drop(test_wells.index)

    # Load pre-trained exponential regression models for each well.
    expreg_models_all = {}
    for model_fn in os.listdir(MODEL_DIR):
        if not model_fn.lower().endswith('.csv'):
            continue
        model_fp = MODEL_DIR / model_fn
        model_name = model_fp.stem
        model_df = pd.read_csv(model_fp)
        expreg_models_all[model_name] = dataframe_to_models(model_df)

    near_wells_params = json_to_nearest_wells_params("nearest_wells_parameters.json")

    exp_reg_params = json_to_exp_regress_params("exp_regress_model_parameters.json")
    # Store results in a dict, keyed by tuple: (near_wells_model_name & expreg_model_name).
    results = {}
    for near_well_model_name in near_wells_params.keys():
        for expreg_model_name in expreg_models_all.keys():
            results[(near_well_model_name, expreg_model_name)] = []
    for near_well_model_name, param_set in near_wells_params.items():
        k = param_set['k']
        distance_weighting = param_set['distance_weighting']
        i = 0
        n = len(test_wells)
        for _, row in test_wells.iterrows():
            row_timestamp = datetime.now()
            i += 1
            api_num = row['API_Label']
            lat_len = row['lateral_length_ft']
            print(f"{near_well_model_name} [{i}/{n}] -- {api_num}", end=" ")
            nearest = find_k_nearest(
                training_wells,
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
                PRODUCTION_RECORDS_DIR / PRODUCTION_CSV_TEMPLATE.format(api_num=api_num),
                parse_dates=['First of Month'],
            )
            prepro = ProductionPreprocessor(actual_prod_raw)
            actual_prod = prepro.preprocess_all()

            for expreg_model_name, expreg_model in expreg_models_all.items():
                if f"{near_well_model_name}__{expreg_model_name}.csv" in existing_comparisons:
                    continue
                print(",", expreg_model_name, end="")
                # Use the original params to ensure we compare apples-to-apples
                # (i.e., use the same actual production window as was used to train).
                params = exp_reg_params[expreg_model_name]
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
                        'predicted': "insufficient data",
                        'actual': "insufficient data",
                        'predicted/actual': None,
                        'prediction_time_cost_sec': (row_timestamp - datetime.now()).microseconds / 1_000_000
                    }
                    results[(near_well_model_name, expreg_model_name)].append(result)
                    continue

                monthly_days = selected_prod_records['calendar_days']
                source_cumulative_days = monthly_days.cumsum()

                # Grab the production models from the k-nearest wells and compute
                # the weighting for each to create a weighted model for our target
                # well.
                nearby_expreg_models = [expreg_model[nearby_api_num] for nearby_api_num in nearest.keys()]
                for m in nearby_expreg_models:
                    pred = m.predict_bbls_per_calendar_day(source_cumulative_days, lateral_length_ft=lat_len)
                distances = list(nearest.values())
                weights = idw_weighting(distances=nearest, power=param_set['distance_weighting'])
                target_model = ExpRegressionModel.weight_models(nearby_expreg_models, weights.values())
                pred_cumulative_days = pd.Series(range(1, sum(monthly_days) + 1))
                predicted_bbls_each_day = target_model.predict_bbls_per_calendar_day(
                    pred_cumulative_days, lateral_length_ft=lat_len)
                predicted_total = sum(predicted_bbls_each_day)
                actual_total = sum(selected_prod_records['Oil Produced'])
                result = {
                    'API_Label': api_num,
                    'months': len(selected_prod_records),
                    'predicted': predicted_total,
                    'actual': actual_total,
                    'predicted/actual': predicted_total / actual_total,
                    'prediction_time_cost_sec': (row_timestamp - datetime.now()).microseconds / 1_000_000
                }
                results[(near_well_model_name, expreg_model_name)].append(result)
            print("")

        # Output our results.
        for model_pair, result in results.items():
            if model_pair[0] != near_well_model_name:
                # Write as we go.
                continue
            fn = f"{model_pair[0]}__{model_pair[1]}.csv"
            results_df = pd.DataFrame(data=result)
            results_df.to_csv(COMPARISON_RESULTS_DIR / fn, index=False)
