"""
Created on Mon July 12 17:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Miscellaneous helper routines.

"""

import ast
import boto3
import io
import numpy as np
import os
import pandas as pd
import re

from arborist.utils.swc_loading import Reader
from botocore import UNSIGNED
from botocore.client import Config
from segmentation_skeleton_metrics.utils.util import parse_cloud_path


class S3RangeReader(io.RawIOBase):
    """
    Seekable, read-only view of an S3 object backed by ranged GETs.

    Lets a large archive be inspected in place, so that reading a ZIP's
    central directory does not require downloading the whole object.

    Parameters
    ----------
    path : str
        Path to the S3 object, in the format "s3://{bucket_name}/{key}".
    """

    def __init__(self, path):
        bucket, key = parse_cloud_path(path)
        self.client = boto3.client(
            "s3", config=Config(signature_version=UNSIGNED)
        )
        self.bucket, self.key = bucket, key
        self.size = self.client.head_object(
            Bucket=bucket, Key=key
        )["ContentLength"]
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            self.pos = offset
        elif whence == os.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def readinto(self, buf):
        if len(buf) == 0 or self.pos >= self.size:
            return 0
        end = min(self.pos + len(buf), self.size) - 1
        body = self.client.get_object(
            Bucket=self.bucket, Key=self.key, Range=f"bytes={self.pos}-{end}"
        )["Body"].read()
        buf[: len(body)] = body
        self.pos += len(body)
        return len(body)


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
    reader = Reader(verbose=False)
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
