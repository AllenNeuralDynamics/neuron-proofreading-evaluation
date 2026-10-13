"""
Created on Mon July 12 17:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for evaluating performance of merge detection models.

"""

import os

import numpy as np
import pandas as pd
from arborist.utils.swc_loading import to_zipped_points
from scipy.spatial import KDTree

from neuron_proofreading_evaluation import visualization as viz

# Same matching radius as the whole-brain merge detection benchmark
MATCH_RADIUS_UM = 25.0

# Predictions further than this from the ground truth skeleton sit on the
# merged-in foreign neurite and are not scored
SKELETON_RADIUS_UM = 15.0

TARGET_RECALLS = (0.97, 0.95)


# --- Performance Metrics ---
def compute_metrics(
    gt_kdtree, pred_sites, max_dist=MATCH_RADIUS_UM, n_total=None
):
    n_gt = len(gt_kdtree.data)
    n_pred = len(pred_sites)

    if n_pred == 0:
        n_tp_prec = n_tp_recall = 0
    else:
        # For each pred, find nearest GT — controls precision
        dd_pred, _ = gt_kdtree.query(pred_sites)
        n_tp_prec = (dd_pred <= max_dist).sum()

        # For each GT, find nearest pred — controls recall (each GT counted
        # once)
        pred_kdtree = KDTree(pred_sites)
        dd_gt, _ = pred_kdtree.query(gt_kdtree.data)
        n_tp_recall = (dd_gt <= max_dist).sum()

    recall = n_tp_recall / (n_gt + 1e-5)
    prec = n_tp_prec / (n_pred + 1e-5)
    f1 = (2 * prec * recall) / (prec + recall + 1e-5)

    result = {
        "# GT Sites": n_gt,
        "# Pred Sites": n_pred,
        "# TP Sites": int(n_tp_recall),
        "Recall": recall,
        "Precision": prec,
        "F1": f1,
    }

    if n_total is not None:
        fp = n_pred - n_tp_prec
        n_neg = n_total - n_gt
        tn = n_neg - fp
        specificity = tn / (n_neg + 1e-5)
        result["Informedness"] = float(recall + specificity - 1)

    return result


def prec_recall_at_threshold(
    gt_df, pred_df, threshold, output_dir=None, preamble="", n_total=None
):
    # Compute result
    gt_kdtree = KDTree(list(gt_df["xyz"].values))
    pred_sites = _pred_sites(pred_df, threshold)
    result = compute_metrics(gt_kdtree, pred_sites, n_total=n_total)

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
        result = compute_metrics(gt_kdtree, _pred_sites(pred_df, t))
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


def filter_near_skeleton(pred_df, skeleton_xyz, max_dist=SKELETON_RADIUS_UM):
    """
    Keeps predictions within "max_dist" of the ground truth skeleton.

    Parameters
    ----------
    pred_df : pandas.DataFrame
        Predictions with an "xyz" column of coordinate tuples.
    skeleton_xyz : numpy.ndarray
        Ground truth skeleton coordinates with shape (n, 3).
    max_dist : float, optional
        Maximum distance (in microns) from the skeleton. Default is
        SKELETON_RADIUS_UM.

    Returns
    -------
    pandas.DataFrame
        Filtered predictions.
    """
    if len(pred_df) == 0 or len(skeleton_xyz) == 0:
        return pred_df
    dd, _ = KDTree(skeleton_xyz).query(np.stack(pred_df["xyz"].values))
    return pred_df[dd <= max_dist].copy()


def is_reachable(gt_xyz, visited_xyz, max_dist=MATCH_RADIUS_UM):
    """
    Flags ground truth sites with a visited (scored) site within "max_dist".
    Sites the search never visits cannot be detected by any model, so recall
    is calibrated to the reachable ones.

    Returns
    -------
    numpy.ndarray
        Boolean mask over the ground truth sites.
    """
    gt_xyz = np.asarray(gt_xyz, dtype=float).reshape(-1, 3)
    if len(gt_xyz) == 0 or len(visited_xyz) == 0:
        return np.zeros(len(gt_xyz), dtype=bool)
    dd, _ = KDTree(visited_xyz).query(gt_xyz)
    return dd <= max_dist


def reachable_recall(gt_xyz, visited_xyz, max_dist=MATCH_RADIUS_UM):
    """
    Counts ground truth sites with a visited (scored) site within
    "max_dist": the recall ceiling the search schedule imposes regardless of
    the model.

    Returns
    -------
    Tuple[int, int]
        Number of reachable ground truth sites and total ground truth sites.
    """
    reachable = is_reachable(gt_xyz, visited_xyz, max_dist)
    return int(reachable.sum()), len(reachable)


def false_negatives(gt_df, pred_df, threshold, max_dist=32):
    """
    Lists ground truth sites without a prediction >= "threshold" within
    "max_dist", along with the highest nearby prediction score (NaN when no
    prediction was scored nearby).
    """
    if len(pred_df):
        pred_tree = KDTree(np.stack(pred_df["xyz"].values))
    else:
        pred_tree = None
    scores = pred_df["Prediction"].values

    rows = list()
    for _, site in gt_df.iterrows():
        if pred_tree:
            idxs = pred_tree.query_ball_point(site["xyz"], max_dist)
        else:
            idxs = []
        max_score = scores[idxs].max() if idxs else np.nan
        if idxs and max_score >= threshold:
            continue
        row = dict(site)
        row["max_score"] = max_score
        row[f"n_pred_within_{max_dist}"] = len(idxs)
        rows.append(row)
    return pd.DataFrame(rows)


def run_evaluation(
    gt_df,
    pred_df,
    output_dir,
    threshold=0.5,
    n_total=None,
    target_recalls=TARGET_RECALLS,
    preamble="",
):
    """
    Runs the full evaluation of merge predictions and writes the results to
    "output_dir": score histogram, precision-recall-F1 curves, per-neuron
    results at "threshold", a performance summary at "threshold" and at the
    threshold reaching each target recall, TP/FP/FN sites and the
    false-negative list at the last target recall.

    Parameters
    ----------
    gt_df : pandas.DataFrame
        Ground truth sites with columns "unique_merge_id", "cell_id", "xyz".
    pred_df : pandas.DataFrame
        Predictions with columns "xyz" and "Prediction".
    output_dir : str
        Directory results are written to.
    threshold : float, optional
        Decision threshold. Default is 0.5.
    n_total : int, optional
        Number of sites scored by the search, used for informedness. Default
        is None.
    target_recalls : Tuple[float], optional
        Recall levels to report operating points at. Default is
        TARGET_RECALLS.
    preamble : str, optional
        Text written at the top of the performance summary. Default is "".

    Returns
    -------
    Dict[str, float]
        Thresholds reported in the performance summary, keyed by label.
    """
    os.makedirs(output_dir, exist_ok=True)
    viz.plot_predictions(
        pred_df["Prediction"].values,
        threshold=threshold,
        output_path=os.path.join(
            output_dir, "model_predictions_histogram.png"
        ),
    )
    prec_recall_curve(gt_df, pred_df, output_dir)
    prec_recall_per_neuron(gt_df, pred_df, threshold, output_dir)
    curve_df = pd.read_csv(
        os.path.join(output_dir, "results_varying_threshold.csv")
    )

    thresholds = {f"t={threshold}": threshold}
    for target in target_recalls:
        # Highest swept threshold that actually reaches the target, so the
        # reported recall is never below the label
        reached = curve_df.loc[curve_df["Recall"] >= target, "Threshold"]
        if len(reached):
            t = float(reached.max())
        else:
            t = threshold_at_recall(curve_df, target_recall=target)
        thresholds[f"t={t:.3f} (Recall>={target:.0%})"] = t

    n_neurons = gt_df["cell_id"].nunique()
    summary_path = os.path.join(output_dir, "performance_summary.txt")
    with open(summary_path, "w") as f:
        f.write(preamble)
        for label, t in thresholds.items():
            result = prec_recall_at_threshold(
                gt_df, pred_df, t, n_total=n_total
            )
            n_gt = result["# GT Sites"]
            n_pred = result["# Pred Sites"]
            n_tp = result["# TP Sites"]
            result["Detected Sites/Neuron"] = n_pred / n_neurons
            result["Merges/Neuron Before"] = n_gt / n_neurons
            result["Merges/Neuron"] = (n_gt - n_tp) / n_neurons

            f.write(f"[{label}]\n")
            for key, value in result.items():
                if isinstance(value, float):
                    f.write(f"   {key}: {value:.4f}\n")
                else:
                    f.write(f"   {key}: {value}\n")
            f.write("\n")

    t_last = list(thresholds.values())[-1]
    save_sites(gt_df, pred_df, t_last, output_dir)
    fn_df = false_negatives(gt_df, pred_df, t_last)
    fn_df.to_csv(
        os.path.join(output_dir, "false_negatives.tsv"), sep="\t", index=False
    )
    return thresholds


# --- Save Results ---
def save_sites(gt_df, pred_df, threshold, output_dir, max_dist=32):
    # Get sites
    gt_sites = np.array(list(gt_df["xyz"].values))
    gt_kdtree = KDTree(gt_sites)
    pred_sites = _pred_sites(pred_df, threshold)

    if len(pred_sites):
        dd, _ = gt_kdtree.query(pred_sites)
        tp_sites = pred_sites[dd <= max_dist]
        fp_sites = pred_sites[dd > max_dist]
    else:
        tp_sites = fp_sites = pred_sites

    if len(tp_sites):
        dd, _ = KDTree(tp_sites).query(gt_sites)
        fn_sites = gt_sites[dd > max_dist]
    else:
        fn_sites = gt_sites

    # Save sites
    sites_path = os.path.join(output_dir, "site_evaluation.zip")
    if os.path.exists(sites_path):
        os.remove(sites_path)
    save_points(sites_path, tp_sites, "0.0 1.0 0.0", "true_positive")
    save_points(sites_path, fp_sites, "1.0 0.0 0.0", "false_positive")
    save_points(sites_path, fn_sites, "1.0 1.0 1.0", "false_negative")


def save_points(zip_path, pts, color, prefix):
    if len(pts) == 0:
        return
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


# --- Helpers ---
def _pred_sites(pred_df, threshold):
    pred_sites = pred_df.loc[pred_df["Prediction"] >= threshold, "xyz"]
    if len(pred_sites):
        return np.stack(pred_sites)
    return np.empty((0, 3), dtype=float)
