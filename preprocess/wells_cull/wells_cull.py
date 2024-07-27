"""
Load and integrate well data (SHL and BHL datasets), and cull it to the
desired relevant wells (limited to wells found in the BHL dataset --
i.e., horizontal wells).

Data sources:
SHL data:  https://ecmc.state.co.us/documents/data/downloads/gis/WELLS_SHP.ZIP
(Extract ``Wells.dbf`` and save it as a ``.csv``.)

BHL data:  https://ecmc.state.co.us/documents/data/downloads/gis/DIRECTIONAL_BOTTOMHOLE_LOCATIONS_SHP.ZIP
(Extract ``Directional_Bottomhole_Locations.dbf`` and save it as a ``.csv``.)
"""

import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import dotenv

__all__ = [
    'load_raw_well_data',
    'cull_by_county_code',
    'cull_by_spud_date',
    'cull_by_well_class',
    'default_load_and_cull',
]

dotenv.load_dotenv()
RAW_DATA_DIR = Path(os.getenv('RAW_DATA_DIR'))
WORKING_DATA_DIR = Path(os.getenv('WORKING_DATA_DIR'))

bhl_cols_to_drop = [
    'Well_Name',
    'Field_Name',
    'Utm_X',
    'Utm_Y',
]
bhl_cols_to_rename = {
    'Lat': 'lat_bhl',
    'Long': 'long_bhl'
}
shl_cols_to_drop = [
    'API',
    'Operator',
    'Well_Num',
    'Well_Name',
    'Well_Title',
    'Field_Code',
    'Field_Name',
    'Utm_X',
    'Utm_Y',
]
shl_cols_to_rename = {
    'Latitude': 'lat_shl',
    'Longitude': 'long_shl'
}
default_api_counties_to_keep = [
    # 1,  # Adams County
    123,  # Weld County
]
default_well_classes_to_keep = [
    'OW',
]
default_shl_csv = RAW_DATA_DIR / r'Wells.csv'
default_bhl_csv = RAW_DATA_DIR / r'Directional_Bottomhole_locations.csv'
default_earliest_spud_date = date(2005, 1, 1)
default_latest_spud_date = date(2019, 12, 31)
default_out_fp = WORKING_DATA_DIR / 'basic_well_data.csv'


def load_raw_well_data(
        shl_csv: Path = default_shl_csv,
        bhl_csv: Path = default_bhl_csv,
) -> pd.DataFrame:
    """
    Load and integrate the SHL and BHL datasets into a combined well
    dataframe.
    :param shl_csv: Filepath to the ``Wells.csv`` file (containing the
     SHL data provided by the ECMC).
    :param bhl_csv: Filepath to the ``Directional_Bottomhole_Locations.csv``
     file (containing the BHL data provided by the ECMC).
    :return: A dataframe of the combined well data.
    """
    bhl = pd.read_csv(bhl_csv)
    bhl.drop(columns=bhl_cols_to_drop, inplace=True)
    bhl.rename(columns=bhl_cols_to_rename, inplace=True)
    # Strip superfluous `-00` at end of API number.
    bhl['API_Label'] = bhl['API_Label'].apply(lambda s: s[:-3])

    shl = pd.read_csv(shl_csv, parse_dates=['Spud_Date', 'Stat_Date'])
    shl.drop(columns=shl_cols_to_drop, inplace=True)
    shl.rename(columns=shl_cols_to_rename, inplace=True)
    shl['Spud_Date'] = pd.to_datetime(shl['Spud_Date'], errors='coerce')

    wells = bhl.merge(
        shl,
        on='API_Label',
        how='left',
        suffixes=('_bhl', '_shl'),
    )
    return wells


def cull_by_county_code(wells: pd.DataFrame, keep_counties: list[int]) -> pd.DataFrame:
    """
    Cull wells by county code (as encoded in the API number) -- i.e.,
    ints ``1`` to ``123``.
    :param wells: A dataframe of the loaded and integrated well data
     (SHL + BHL).
    :param keep_counties: A list of county codes to keep, such as ``[1, 5, 123]``.
    :return: A new dataframe limited to the desired wells.
    """
    return wells.drop(wells[~wells['API_County'].isin(keep_counties)].index)


def cull_by_spud_date(
        wells: pd.DataFrame,
        earliest=date(2005, 1, 1),
        latest=date(2100, 12, 31),
        drop_missing: bool = True,
) -> pd.DataFrame:
    """
    Cull wells by minimum spud date. (Defaults to culling wells spudded
    prior to 1/1/2005).

    :param wells: A dataframe of the loaded and integrated well data
     (SHL + BHL).
    :param earliest: A ``datetime.date`` object representing the
     earliest spud date to keep. (Defaults to 1/1/2005.)
    :param latest: A ``datetime.date`` object representing the latest
     spud date to keep. (Defaults to 12/31/2100.)
    :param drop_missing: Whether to drop any rows that are missing a
     (valid) spud date. (Defaults to ``True``.)
    :return: A new dataframe limited to the desired wells.
    """
    if drop_missing:
        wells = wells.drop(wells[wells['Spud_Date'].isnull()].index)
    mask = wells[
        (wells['Spud_Date'] < datetime(earliest.year, earliest.month, earliest.day))
        | (wells['Spud_Date'] > datetime(latest.year, latest.month, latest.day))
        ]
    return wells.drop(mask.index)


def cull_by_well_class(wells: pd.DataFrame, keep_well_classes: list[str]) -> pd.DataFrame:
    """
    Cull wells by county code (as encoded in the API number), i.e., ints
    ``1`` to ``123``.
    :param wells: A dataframe of the loaded and integrated well data
     (SHL + BHL).
    :param keep_well_classes: A list of well classes keep, such as ``[OW, GW]``.
    :return: A new dataframe limited to the desired wells.
    """
    return wells.drop(wells[~wells['Well_Class'].isin(keep_well_classes)].index)


def default_load_and_cull() -> pd.DataFrame:
    """
    Load and cull wells with configured defaults.
    :return: A dataframe with the loaded, integrated, and culled well
     records.
    """
    wells = load_raw_well_data(RAW_DATA_DIR)
    wells = cull_by_county_code(wells, keep_counties=default_api_counties_to_keep)
    wells = cull_by_spud_date(
        wells,
        earliest=default_earliest_spud_date,
        latest=default_latest_spud_date,
        drop_missing=True)
    wells = cull_by_well_class(wells, keep_well_classes=default_well_classes_to_keep)
    return wells
