import os
from pathlib import Path

import dotenv

from .shl_bhl.lat_long import (
    fill_lat_long_midpoints,
    fill_lateral_distances,
)
from preprocess.scrape_cleanup.wells_cull import (
    default_load_and_cull,
    cull_by_twprge,
)

dotenv.load_dotenv()
RAW_DATA_DIR = Path(os.getenv('RAW_DATA_DIR'))
WORKING_DATA_DIR = Path(os.getenv('WORKING_DATA_DIR'))
WORKING_DATA_FN = Path(os.getenv('WORKING_DATA_FN'))

__all__ = [
    'clean_working_dataset',
]


def clean_working_dataset():
    wells = default_load_and_cull()
    keep_twprge = [
        '7n60w',
        '7n59w',
        '8n59w',
        '8n60w',
        '9n58w',
        '9n59w',
        '9n60w',
        '10n58w',
        '10n57w',
    ]
    wells = cull_by_twprge(wells, keep_twprge)
    wells = fill_lat_long_midpoints(wells)
    wells = fill_lateral_distances(wells)
    wells.to_csv(WORKING_DATA_DIR / WORKING_DATA_FN, index=False)


if __name__ == '__main__':
    clean_working_dataset()
