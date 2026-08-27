"""
Created on Wed Aug 19 09:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Class for managing the S3 directory structure of a pipeline result.

"""

import boto3
import os
import re

from segmentation_skeleton_metrics.utils.util import parse_cloud_path


class ResultsManager:
    """
    Discovers and manages paths for a pipeline results directory in S3.

    Step directories matching step{N}_* are sorted by step number and
    classified as "merge" or "split" based on their name. All steps get a
    directory path in step_dirs; merge steps additionally get a swcs.zip
    path in step_swcs_paths.
    """

    def __init__(self, s3_prefix, brain_id, segmentation_id):
        self.s3_prefix = s3_prefix.rstrip("/") + "/"
        self.brain_id = brain_id
        self.segmentation_id = segmentation_id
        self.eval_steps = ["original"]
        self.step_types = {"original": "original"}
        self.step_dirs = {}
        self.step_swcs_paths = {"original": os.path.join(s3_prefix.rstrip("/"), "original_swcs")}
        self._get_steps()

    def _get_steps(self):
        # Create S3 reader
        bucket, prefix = parse_cloud_path(self.s3_prefix)
        s3 = boto3.client("s3")
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")

        # Search for step directories
        discovered = []
        for cp in resp.get("CommonPrefixes", []):
            dirname = cp["Prefix"].rstrip("/").split("/")[-1]
            m = re.match(r"step(\d+)_", dirname)
            if m:
                discovered.append((int(m.group(1)), dirname))
        discovered.sort()

        # Add steps
        for _, dirname in discovered:
            self._add_step(dirname)

    def _add_step(self, dirname):
        dir_path = os.path.join(self.s3_prefix, dirname)
        self.eval_steps.append(dirname)
        self.step_dirs[dirname] = dir_path
        self.step_swcs_paths[dirname] = os.path.join(dir_path, "swcs.zip")
        self.step_types[dirname] = "split" if "split_correction" in dirname else "merge"

    def get_labels_path(self):
        return os.path.join(self.s3_prefix, "segment_ids.txt")
