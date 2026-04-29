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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neuron_proofreading_evaluation.proofread_splits import (
    data_handling as data_util,
)


# --- Precision-Recall Curves ---
def compute_precision_recall(
    gt_graphs, fragment_graphs, labels, proposal_df, dt=0.02
):
    results_df = create_thresholded_results_df(dt=dt)
    for t in tqdm(results_df.index, desc="Thresholded Precision-Recall"):
        # Compile proposals
        proposal_df_t = data_util.get_subdf(proposal_df, False, t)
        skip_merge_cnt = t < 0.29

        # Compute metrics
        n_splits, n_merges = count_splits_and_merges(
            deepcopy(gt_graphs),
            deepcopy(fragment_graphs),
            labels,
            proposal_df_t,
            skip_merge_cnt=skip_merge_cnt,
        )

        results_df.loc[t, "# Splits"] = n_splits
        results_df.loc[t, "# Merges"] = n_merges

    return compute_precision_recall_from_df(results_df)


def compute_multiround_precision_recall(
    gt_graphs, fragment_graphs, labels, proposal_df_list
):
    n_rounds = len(proposal_df_list)
    results_df = create_multiround_results_df(n_rounds)
    for k in tqdm(np.arange(n_rounds + 1), desc="Precision-Recall-F1"):
        # Compile proposals
        proposal_df_k = proposal_df_list[0:k]
        if k > 0:
            proposal_df_k = pd.concat(proposal_df_k, ignore_index=True)

        # Compute metrics
        n_splits, n_merges = count_splits_and_merges(
            deepcopy(gt_graphs),
            deepcopy(fragment_graphs),
            labels,
            proposal_df_k,
        )

        results_df.loc[k, "# Splits"] = n_splits
        results_df.loc[k, "# Merges"] = n_merges

    # Compute precision-recall-f1
    results_df.loc[np.inf, "# Splits"] = results_df.loc[0, "# Splits"]
    results_df.loc[np.inf, "# Merges"] = results_df.loc[0, "# Merges"]
    results_df = compute_precision_recall_from_df(results_df)
    return results_df


def compute_precision_recall_from_df(results_df):
    initial_merges = results_df.loc[np.inf, "# Merges"]
    initial_splits = results_df.loc[np.inf, "# Splits"]
    for i in results_df.index:
        tp = initial_splits - results_df.loc[i, "# Splits"]
        fp = results_df.loc[i, "# Merges"] - initial_merges

        precision = 1 - fp / (fp + tp + 1e-5)
        recall = tp / (initial_splits + 1e-5)
        f1 = 2 * precision * recall / (precision + recall)

        results_df.loc[i, "Precision"] = round(precision, 4)
        results_df.loc[i, "Recall"] = round(recall, 4)
        results_df.loc[i, "F1"] = round(f1, 4)
    return results_df


def count_splits_and_merges(
    gt_graphs, fragment_graphs, labels, proposals_df, skip_merge_cnt=False
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
    if skip_merge_cnt:
        n_merges = np.nan
    else:
        merge_count_metric = MergeCountMetric(verbose=False)
        merge_cnts = merge_count_metric(gt_graphs, fragment_graphs)
        n_merges = merge_cnts["# Merges"].sum()

    return n_splits, n_merges


def save_precision_recall_curve(df, path, show_midpoint=False, title=""):
    plt.figure()
    for t in ["Precision", "Recall", "F1"]:
        plt.plot(df.index, df[t], label=t, linewidth=3, zorder=3)

    if show_midpoint:
        x = (len(df) - 1) / 2
        plt.axvline(x, linestyle="dotted", linewidth=1.1, color="k", zorder=3)

    plt.xlabel(df.index.name, fontsize=13)
    plt.ylabel("Score", fontsize=13)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, zorder=0)

    plt.savefig(path, dpi=300, bbox_inches="tight")


# --- Helpers ---
def create_thresholded_results_df(dt=0.02):
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
