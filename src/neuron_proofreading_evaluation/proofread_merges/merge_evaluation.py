"""
Created on Mon July 12 17:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for evaluating performance of merge detection models.

"""

from scipy.spatial import KDTree

import numpy as np
import os
import pandas as pd

from neuron_proofreading_evaluation import visualization as viz
from arborist.utils.swc_loading import to_zipped_points


# --- Performance Metrics ---
def compute_metrics(gt_kdtree, pred_sites, max_dist=20):
    n_gt = len(gt_kdtree.data)
    n_pred = len(pred_sites)

    # For each pred, find nearest GT — controls precision
    dd_pred, _ = gt_kdtree.query(pred_sites)
    n_tp_prec = (dd_pred <= max_dist).sum()

    # For each GT, find nearest pred — controls recall (each GT counted once)
    pred_kdtree = KDTree(pred_sites)
    dd_gt, _ = pred_kdtree.query(gt_kdtree.data)
    n_tp_recall = (dd_gt <= max_dist).sum()

    recall = n_tp_recall / (n_gt + 1e-5)
    prec = n_tp_prec / (n_pred + 1e-5)
    f1 = (2 * prec * recall) / (prec + recall + 1e-5)

    return {
        "# GT Sites": n_gt,
        "# Pred Sites": n_pred,
        "# TP Sites": int(n_tp_recall),
        "Recall": recall,
        "Precision": prec,
        "F1": f1,
    }


def prec_recall_at_threshold(
    gt_df, pred_df, threshold, output_dir=None, preamble=""
):
    # Compute result
    gt_kdtree = KDTree(list(gt_df["xyz"].values))
    pred_sites = pred_df.loc[pred_df["Prediction"] >= threshold, "xyz"]
    pred_sites = np.stack(pred_sites)
    result = compute_metrics(gt_kdtree, pred_sites)

    # Save results
    if output_dir:
        output_path = os.path.join(output_dir, "performance_summary.txt")
        write_results(result, output_path, preamble=preamble)
    return result


def prec_recall_curve(gt_df, pred_df, output_dir, dt=0.01):
    # Compute performance metrics for varying thresholds
    gt_kdtree = KDTree(list(gt_df["xyz"].values))
    results = list()
    for t in np.arange(0, 1 + dt, dt):
        # Get predicted sites
        pred_sites = pred_df.loc[pred_df["Prediction"] >= t, "xyz"]
        if len(pred_sites):
            pred_sites = np.stack(pred_sites)
        else:
            pred_sites = np.empty((0, 3), dtype=float)

        # Compute metrics
        result = compute_metrics(gt_kdtree, pred_sites)
        result["Threshold"] = t
        results.append(result)

    # Save results
    path = os.path.join(output_dir, "results_varying_threshold.csv")
    results_df = pd.DataFrame(results).set_index("Threshold")
    results_df.to_csv(path)

    output_path = os.path.join(output_dir, "prec_recall_f1_curves.png")
    viz.plot_precision_recall_f1(results_df, output_path=output_path)
    return results


def prec_recall_per_neuron(gt_df, pred_df, threshold, output_dir):
    results = list()
    for neuron_id, neuron_gt_df in gt_df.groupby("cell_id"):
        result = prec_recall_at_threshold(neuron_gt_df, pred_df, threshold)
        result["Neuron ID"] = neuron_id
        results.append(result)

    # Save results
    path = os.path.join(output_dir, "results_per_neuron.csv")
    results_df = pd.DataFrame(results).set_index("Neuron ID")
    results_df.to_csv(path)
    return results


def threshold_at_recall(df, target_recall=0.90):
    """
    Gets the threshold at which recall first drops to "target_recall".
    """
    # Extract sub-dataframe
    subdf = df[["Threshold", "Recall"]].sort_values("Threshold")
    subdf = subdf.reset_index(drop=True)

    is_above = target_recall > subdf["Recall"].iloc[0]
    is_below = target_recall < subdf["Recall"].iloc[-1]
    if is_above or is_below:
        return 0.2

    # First row where recall <= target_recall
    idx = (subdf["Recall"] <= target_recall).idxmax()
    if idx == 0:
        return subdf["Threshold"].iloc[0]

    t0, r0 = subdf["Threshold"].iloc[idx - 1], subdf["Recall"].iloc[idx - 1]
    t1, r1 = subdf["Threshold"].iloc[idx], subdf["Recall"].iloc[idx]

    if r0 == r1:
        return t0

    # Interpolation between (t0, r0) and (t1, r1)
    frac = (r0 - target_recall) / (r0 - r1)
    return t0 + frac * (t1 - t0)


# --- Save Results ---
def save_sites(gt_df, pred_df, threshold, output_dir, max_dist=32):
    # Get sites
    gt_sites = np.array(list(gt_df["xyz"].values))
    gt_kdtree = KDTree(gt_sites)
    pred_sites = pred_df.loc[pred_df["Prediction"] >= threshold, "xyz"]
    pred_sites = np.stack(pred_sites)

    dd, _ = gt_kdtree.query(pred_sites)
    tp_sites = pred_sites[dd <= max_dist]
    fp_sites = pred_sites[dd > max_dist]

    dd, _ = KDTree(tp_sites).query(gt_sites)
    fn_sites = gt_sites[dd > max_dist]

    # Save sites
    sites_path = os.path.join(output_dir, "site_evaluation.zip")
    save_points(sites_path, tp_sites, "0.0 1.0 0.0", "true_positive")
    save_points(sites_path, fp_sites, "1.0 0.0 0.0", "false_positive")
    save_points(sites_path, fn_sites, "1.0 1.0 1.0", "false_negative")


def save_points(zip_path, pts, color, prefix):
    to_zipped_points(
        zip_path, pts, color=color, prefix=prefix, radius=10, write_mode="a"
    )


def write_results(results, output_path, preamble=""):
    """
    Writes a results dictionary to a text file, one "key: value" pair per
    line.

    Parameters
    ----------
    results : Dict[str, float]
        Dictionary of result names to values.
    output_path : str
        Path to write the text file to.
    """
    with open(output_path, "w") as f:
        f.write(preamble)
        for key, value in results.items():
            if isinstance(value, float):
                f.write(f"   {key}: {value:.4f}\n")
            else:
                f.write(f"   {key}: {value}\n")
