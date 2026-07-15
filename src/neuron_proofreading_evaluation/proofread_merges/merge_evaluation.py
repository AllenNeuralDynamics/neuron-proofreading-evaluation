"""
Created on Mon July 12 17:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for evaluating performance of merge detection models.

"""

from scipy.spatial import KDTree
from tqdm import tqdm

import numpy as np
import os
import pandas as pd

from neuron_proofreading_evaluation import visualization as viz
from neuron_proofreader.utils import swc_util


# --- Performance Metrics ---
def compute_metrics(gt_kdtree, pred_sites, max_dist=32):
    # Get GT and Pred overlap
    dd, _ = gt_kdtree.query(pred_sites)
    n_gt = len(gt_kdtree.data)
    n_pred = len(pred_sites)
    n_tp = len(pred_sites[dd <= max_dist])

    # Compute metrics
    recall = n_tp / (n_gt + 1e-5)
    prec = n_tp / (n_pred + 1e-5)
    f1 = (2 * prec * recall) / (prec + recall + 1e-5)

    # Store results
    result = {
        "# GT Sites": n_gt,
        "# Pred Sites": n_pred,
        "# TP Sites": n_tp,
        "Recall": recall,
        "Precision": prec,
        "F1": f1,
    }
    return result


def prec_recall_at_threshold(
    gt_df, pred_df, threshold, output_dir, preamble=""
):
    # Compute result
    gt_kdtree = KDTree(list(gt_df["xyz"].values))
    pred_sites = pred_df.loc[pred_df["Prediction"] > threshold, "xyz"]
    pred_sites = np.stack(pred_sites)
    result = compute_metrics(gt_kdtree, pred_sites)

    # Save results
    output_path = os.path.join(output_dir, "performance_summary.txt")
    write_results(result, output_path, preamble=preamble)


def prec_recall_curve(gt_df, pred_df, output_dir, dt=0.01):
    # Compute performance metrics for varying thresholds
    gt_kdtree = KDTree(list(gt_df["xyz"].values))
    results = list()
    for t in tqdm(np.arange(0, 1 + dt, dt), desc="Varying Threshold"):
        # Get predicted sites
        pred_sites = pred_df.loc[pred_df["Prediction"] > t, "xyz"]
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
    print(results_df)

    output_path = os.path.join(output_dir, "prec_recall_f1_curves.png")
    viz.plot_precision_recall_f1(results_df, output_path=output_path)


def prec_recall_per_neuron(gt_df, pred_df, output_dir):
    pass


# --- Save Results ---
def save_sites(gt_df, pred_df, threshold, output_dir, max_dist=32):
    # Get sites
    gt_sites = np.array(list(gt_df["xyz"].values))
    gt_kdtree = KDTree(gt_sites)
    pred_sites = pred_df.loc[pred_df["Prediction"] > threshold, "xyz"]
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
    swc_util.write_points(
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
