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

import numpy as np
import pandas as pd

from neuron_proofreading_evaluation.proofread_splits import (
    data_handling as data_util,
)


# --- Precision-Recall Curves ---
def compute_precision_recall(
    gt_graphs, fragment_graphs, labels, csv_paths
):
    # Initializations
    n_rounds = len(csv_paths)
    proposal_df_list = data_util.load_proposal_df(csv_paths)
    results_df = create_multiround_results_df(n_rounds)

    # Evaluate
    for k in tqdm(np.arange(n_rounds + 1), desc="Precision-Recall-F1":
        # Compile proposals
        proposal_df_k = proposal_df_list[0:k]
        if k > 0:
            proposal_df_k = pd.concat(proposal_df_k, ignore_index=True)

        # Compute metrics
        n_splits, n_merges = count_splits_and_merges(
            deepcopy(gt_graphs),
            deepcopy(fragment_graphs),
            labels,
            proposal_df_k
        )

        results_df.loc[k, "# Splits"] = n_splits
        results_df.loc[k, "# Merges"] = n_merges

    compute_precision_recall_from_df(results_df)
    return results_df


def compute_precision_recall_from_df(results_df):
    initial_merges = results_df.loc[0, "# Merges"]
    initial_splits = results_df.loc[0, "# Splits"]
    for i in results_df.index:
        tp = initial_splits - results_df.loc[i, "# Splits"]
        fp = results_df.loc[i, "# Merges"] - initial_merges

        precision = 1 - fp / (fp + tp + 1e-5)
        recall = tp / (initial_splits + 1e-5)
        f1 = 2 * precision * recall / (precision + recall)

        results_df.loc[i, "Precision"] = round(precision, 4)
        results_df.loc[i, "Recall"] = round(recall, 4)
        results_df.loc[i, "F1"] = round(f1, 4)


def count_splits_and_merges(
    gt_graphs, fragment_graphs, labels, proposals_df
):
    # Relabel data
    if len(proposals_df) > 0:
        gt_graphs, fragment_graphs = data_util.apply_label_mapping_to_graphs(
            gt_graphs,
            fragment_graphs,
            labels,
            proposals_df,
        )

    # Count splits
    split_count_metric = SplitCountMetric(verbose=False)
    split_cnts = split_count_metric(gt_graphs)
    n_splits = split_cnts["# Splits"].sum()

    # Count merges
    merge_count_metric = MergeCountMetric(verbose=False)
    merge_cnts = merge_count_metric(gt_graphs, fragment_graphs)
    n_merges = merge_cnts["# Merges"].sum()

    return n_splits, n_merges


# --- Helpers ---
def create_results_df(dt=0.02):
    # Create empty dataframe
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
    results_df = results_df.set_index("Threshold")

    # Add row for inital merge and split counts
    final_row = pd.DataFrame(index=[np.inf], columns=columns[1:])
    results_df = pd.concat([results_df, final_row])
    return results_df


def create_multiround_results_df(n_rounds):
    # Create empty dataframe
    columns = [
        "Round",
        "# Merges",
        "# Splits",
        "Precision",
        "Recall",
        "F1",
    ]
    results_df = pd.DataFrame(columns=columns)
    results_df["Round"] = np.arange(n_rounds + 1).astype(int)
    results_df = results_df.set_index("Round")

    # Add row for inital merge and split counts
    final_row = pd.DataFrame(index=[np.inf], columns=columns[1:])
    results_df = pd.concat([results_df, final_row])
    return results_df
