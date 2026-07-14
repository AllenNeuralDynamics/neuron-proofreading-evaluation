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


# --- Performance Metrics ---
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


def compute_metrics(gt_kdtree, pred_sites, max_dist=32):
    # Get GT and Pred overlap
    dd, _ = gt_kdtree.query(pred_sites)
    n_gt = len(gt_kdtree.data)
    n_pred = len(pred_sites)
    n_tp = len(pred_sites[dd <= max_dist])

    # Compute metrics
    recall = n_tp / (n_gt + 1e-5)
    prec= n_tp / (n_pred + 1e-5)
    f1 = (2 * prec * recall) / (prec + recall + 1e-5)

    # Store results
    result = {
        "# GT Sites": n_gt,
        "# Pred Sites": n_pred,
        "# TP Sites": n_tp,
        "Recall": recall,
        "Precision": prec,
        "F1": f1
    }
    return result


# --- Save Results ---
def save_sites(gt_id, gt_sites, pred_sites):
    # Get sites
    dd, _ = KDTree(gt_sites).query(pred_sites)
    tp_sites = pred_sites[dd <= 32]
    fp_sites = pred_sites[dd > 32]

    dd, _ = KDTree(tp_sites).query(gt_sites)
    fn_sites = gt_sites[dd > 32]

    # Save sites
    sites_path = os.path.join(output_dir, f"results-{gt_id}.zip")
    save_points(sites_path, tp_sites, "0.0 1.0 0.0", "true_positive")
    save_points(sites_path, fp_sites, "1.0 0.0 0.0", "false_positive")
    save_points(sites_path, fn_sites, "1.0 1.0 1.0", "false_negative")


def save_points(zip_path, pts, color, prefix):
    swc_util.write_points(
        zip_path,
        pts,
        color=color,
        prefix=prefix,
        radius=10,
        write_mode="a"
    )
