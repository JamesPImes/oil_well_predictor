
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import dotenv

from preprocess import ProductionPreprocessor
from utils import get_prod_window
from model import ExpRegressionModel, dataframe_to_models

dotenv.load_dotenv()

PRODUCTION_RECORDS_DIR = Path(os.getenv("PRODUCTION_RECORDS_DIR"))
PRODUCTION_CSV_TEMPLATE = "{api_num}_production_data.csv"


def get_and_prep_production_records(api_num: str) -> pd.DataFrame:
    fp = PRODUCTION_RECORDS_DIR / PRODUCTION_CSV_TEMPLATE.format(api_num=api_num)
    production_records = pd.read_csv(fp, parse_dates=['First of Month'])
    prepro = ProductionPreprocessor(production_records)
    cleaned_prod = prepro.preprocess_all()
    return get_prod_window(cleaned_prod)


def load_exp_regress_models():
    ...

def graph_production(prod_records, model: ExpRegressionModel, lat_len: float = None):
    x = prod_records['calendar_days'].cumsum()
    y = prod_records['bbls_per_calendar_day']
    days = pd.Series(range(max(x)))
    predicted_bbls = model.predict_bbls_per_calendar_day(days, lateral_length_ft=lat_len)
    ax = plt.subplot()
    ax.set_title('Daily Production')
    ax.set_xlabel('Days')
    ax.set_ylabel('BBLs')
    ax.plot(x, y, label='Actual')
    ax.plot(days, predicted_bbls, label='Predicted')
    ax.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    ...
