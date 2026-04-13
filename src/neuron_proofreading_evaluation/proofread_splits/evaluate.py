"""
Created on Tue Mar 10 16:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for evaluating the split correction pipeline

"""

from copy import deepcopy
from segmentation_skeleton_metrics.skeleton_metrics import (
    MergeCountMetric,
    SplitCountMetric,
)
from tqdm import tqdm

import numpy as np
import pandas as pd

from neuron_proofreading_evaluation.proofread_splits import (
    data_handling as data_util,
)


# --- Precision-Recall Curves ---
def compute_precision_recall_from_df(results_df):
    initial_merges = results_df.loc[1.0, "# Merges"]
    initial_splits = results_df.loc[1.0, "# Splits"]
    for t in results_df.index:
        tp = initial_splits - results_df.loc[t, "# Splits"]
        fp = results_df.loc[t, "# Merges"] - initial_merges

        precision = 1 - fp / (fp + tp + 1e-5)
        recall = tp / (initial_splits + 1e-5)
        f1 = 2 * precision * recall / (precision + recall)

        results_df.loc[t, "Precision"] = round(precision, 4)
        results_df.loc[t, "Recall"] = round(recall, 4)
        results_df.loc[t, "F1"] = round(f1, 4)


def compute_splits_and_merges(
    gt_graphs, fragment_graphs, labels, proposals_df, results_df
):
    merge_count_metric = MergeCountMetric(verbose=False)
    split_count_metric = SplitCountMetric(verbose=False)
    for t in tqdm(results_df.index, desc="Precision-Recall Curves"):
        # Build graphs
        gt_graphs_t, fragment_graphs_t = data_util.build_graphs_at_threshold(
            t,
            deepcopy(gt_graphs),
            deepcopy(fragment_graphs),
            labels,
            proposals_df,
        )

        # Count merges (if applicable)
        if t > 0.19:
            merge_cnts = merge_count_metric(gt_graphs_t, fragment_graphs_t)
            results_df.loc[t, "# Merges"] = merge_cnts["# Merges"].sum()

        # Count splits
        split_cnts = split_count_metric(gt_graphs_t)
        results_df.loc[t, "# Splits"] = split_cnts["# Splits"].sum()


# --- Helpers ---
def create_results_df(dt=0.05):
    columns = [
        "Threshold",
        "# Merges",
        "# Splits",
        "Precision",
        "Recall",
        "F1",
    ]
    results_df = pd.DataFrame(columns=columns)
    results_df["Threshold"] = np.round(np.arange(0, 1 + dt, dt), decimals=2)
    return results_df.set_index("Threshold")
