"""
Created on Mon July 12 17:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Miscellaneous helper routines.

"""

import ast
import numpy as np
import pandas as pd
import re

from neuron_proofreader.utils import swc_util


def load_sites_df(path):
    """
    Loads a CSV containing site information and parses the "xyz" column.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded dataframe.
    """
    df = pd.read_csv(path)
    df["xyz"] = df["xyz"].apply(ast.literal_eval)
    return df


def load_swc_points(swc_path):
    reader = swc_util.Reader(verbose=False)
    swc_dicts = reader(swc_path)
    return np.array([swc_dict["xyz"] for swc_dict in swc_dicts]).squeeze()


def _cast(value):
    """
    Casts a string value to bool, int, or float where possible, else leaves
    it as a string.

    Parameters
    ----------
    value : str
        Value to be casted.

    Returns
    -------
    value : bool, int, float, or str
        Casted value.
    """
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value
