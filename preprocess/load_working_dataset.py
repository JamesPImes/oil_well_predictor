import os
from pathlib import Path

import dotenv
import pandas as pd

dotenv.load_dotenv()
WORKING_DATA_DIR = Path(os.getenv('WORKING_DATA_DIR'))
WORKING_DATA_FN = Path(os.getenv('WORKING_DATA_FN'))

__all__ = [
    'load_working_dataset',
]


def load_working_dataset(fp: Path = WORKING_DATA_DIR / WORKING_DATA_FN):
    return pd.read_csv(fp, parse_dates=['Spud_Date', 'Stat_Date'])
