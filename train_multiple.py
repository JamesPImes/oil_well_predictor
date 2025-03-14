"""
A script to train all exponential regression models configured in
`exp_regress_model_parameters.json`.
"""

import os
from pathlib import Path

import dotenv
import pandas as pd

from preprocess.prod_records.monthly_prod import ProductionPreprocessor
from model.exp_regress.exp_regress import ExpRegressionModel
from utils.json_to_model_params import json_to_exp_regress_params

dotenv.load_dotenv()
# Location for pre-trained model data.
MODEL_DIR = Path(os.getenv('MODEL_DIR'))
MODEL_TEMPLATE = "{model_name}.csv"
# Location for raw production records.
PRODUCTION_RECORDS_DIR = Path(os.getenv('PRODUCTION_RECORDS_DIR'))
PRODUCTION_CSV_TEMPLATE = "{api_num}_production_data.csv"
WELL_DATA_FP = Path(os.getenv('WELL_DATA_FP'))

if __name__ == '__main__':
    if not WELL_DATA_FP.exists():
        raise RuntimeError(
            "Preprocess well data first (SHL and BHL).\n"
            f"Expected at: {str(WELL_DATA_FP)!r}"
        )
    wells = pd.read_csv(WELL_DATA_FP, parse_dates=['Spud_Date', 'Stat_Date'])
    wells = wells.dropna(subset=['lateral_length_ft'])
    MODEL_DIR.mkdir(exist_ok=True)
    # Load model parameters.
    exp_regress_params = json_to_exp_regress_params('exp_regress_model_parameters.json')
    # For results.
    trained = {model_name: [] for model_name in exp_regress_params.keys()}
    # Check if any models are actually still needed.
    existing_models = os.listdir(MODEL_DIR)
    models_needed = []
    for model_name in exp_regress_params.keys():
        model_fn = MODEL_TEMPLATE.format(model_name=model_name)
        if model_fn not in existing_models:
            models_needed.append(model_name)
    if not models_needed:
        input(
            f"All models have been trained. "
            f"To retrain, delete models in directory:\n"
            f"{MODEL_DIR}"
        )
        exit()

    i = 0
    n = len(wells)
    for _, row in wells.iterrows():
        i += 1
        api_num = row['API_Label']
        lat_len = row['lateral_length_ft']
        print(f"Processing [{i}/{n}] -- {api_num}...", end=" ")
        prod_fp = PRODUCTION_RECORDS_DIR / PRODUCTION_CSV_TEMPLATE.format(api_num=api_num)
        prod_raw = pd.read_csv(prod_fp, parse_dates=['First of Month'])
        preprocessor = ProductionPreprocessor(prod_raw)
        prod = preprocessor.preprocess_all()
        formations = sorted(preprocessor.formations)

        for model_name in models_needed:
            print(",", model_name, end="")
            exp_params = exp_regress_params[model_name]
            model = ExpRegressionModel(lateral_length_ft=lat_len, **exp_params)
            model.train(prod)
            export_attributes = (
                'a',
                'b',
                'min_months',
                'max_months',
                'discard_gaps',
                'lateral_length_ft',
                'has_produced',
                'sufficient_data',
                'actual_months',
            )
            results = {att: getattr(model, att) for att in export_attributes}
            results['API_Label'] = api_num
            results['formations'] = '|'.join(formations)
            trained[model_name].append(results)
        print("")

    # Dump results to csv.
    for model_name, results in trained.items():
        print(f"Saving model {model_name!r}...")
        model_df = pd.DataFrame(data=results)
        if len(model_df) == 0:
            continue
        out_fp_model = MODEL_DIR / MODEL_TEMPLATE.format(model_name=model_name)
        model_df.to_csv(out_fp_model, index=False)
    print("Done.")
