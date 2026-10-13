"""
Created on Thu July 13 18:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for visualizing evaluation results.

"""

import os

import matplotlib.pyplot as plt
import numpy as np
from neuron_proofreader.utils import img_util
from scipy.spatial import KDTree


def plot_precision_recall_f1(
    df, output_path=None, title="Precision-Recall-F1 Curves"
):
    """
    Plots precision, recall, and F1 curves across a threshold sweep.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with "Threshold", "Precision", "Recall", and "F1".
        columns.
    output_path : str, optional
        If provided, the figure is saved to this location. Otherwise, it is
        displayed. Default is None.
    title : str, optional
        Title for the figure. Default is "Precision-Recall-F1 Curves".
    """
    # Set colors
    colors = {
        "Precision": "#3B6FA0",
        "Recall": "#C0392B",
        "F1": "#4E9F5C",
    }
    text_color = "#2B2B2B"

    # Create plot
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for metric, color in colors.items():
        ax.plot(
            df.index,
            df[metric],
            color=color,
            linewidth=2.2,
            label=metric,
            zorder=3,
        )

    # Add axes
    ax.set_xlabel("Threshold", fontsize=11, color=text_color, labelpad=8)
    ax.set_ylabel("Score", fontsize=11, color=text_color, labelpad=8)
    ax.set_title(title, fontsize=14, color=text_color, pad=14)
    ax.set_ylim(0, 1.05)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.tick_params(colors="#555555", labelsize=9.5)
    ax.grid(axis="y", which="major", color="#E0E0E0", linewidth=0.8, zorder=0)

    # Add legend
    legend = ax.legend(
        edgecolor="none",
        facecolor="white",
        framealpha=0.5,
        frameon=True,
        fontsize=9.5,
        loc="upper right",
    )
    for text in legend.get_texts():
        text.set_color(text_color)

    plot_result(output_path=output_path)


def plot_predictions(
    pred, threshold=0.5, bins=30, output_path=None, title="Model Predictions"
):
    """
    Plots a log-scaled histogram of a 1D array of prediction scores.

    Parameters
    ----------
    pred : numpy.ndarray
        1D array of prediction scores (e.g. probabilities in [0, 1]).
    threshold : float, optional
        Decision threshold to overlay on the plot. Default is 0.5.
    bins : int, optional
        Number of histogram bins. Default is 30.
    output_path : str, optional
        If provided, the figure is saved to this location. Otherwise, it is
        displayed. Default is None.
    title : str, optional
        Title for the figure. Default is "Model Predictions".
    """
    # Colors
    bar_color = "#3B6FA0"
    threshold_color = "#C0392B"
    text_color = "#2B2B2B"

    # Create plot
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Add data
    pred = np.asarray(pred).ravel()
    ax.hist(
        pred,
        bins=bins,
        color=bar_color,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    ax.axvline(
        threshold,
        color=threshold_color,
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold = {threshold}",
        zorder=4,
    )

    # Set axes
    ax.set_yscale("log")
    ax.set_xlabel("Score", fontsize=11, color=text_color, labelpad=8)
    ax.set_ylabel("Count", fontsize=11, color=text_color, labelpad=8)
    ax.set_title(title, fontsize=14, color=text_color, pad=14)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.tick_params(colors="#555555", labelsize=9.5)
    ax.grid(axis="y", which="major", color="#E0E0E0", linewidth=0.8, zorder=0)

    # Add legend
    legend = ax.legend(frameon=False, fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(text_color)

    n_pos = int((pred >= threshold).sum())
    fig.text(
        0.5,
        -0.03,
        f"n = {len(pred)}   |   mean = {pred.mean():.3f}   |   "
        f"above threshold = {n_pos} ({n_pos / len(pred):.1%})",
        ha="center",
        fontsize=9.5,
        color="#666666",
    )

    plot_result(output_path=output_path)


# --- Helpers ---
def plot_result(output_path=None):
    """
    Displays or saves the current Matplotlib figure.

    Parameters
    ----------
    output_path : str, optional
        If provided, the figure is saved to this location. Otherwise, it is
        displayed. Default is None.
    """
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def save_gt_site_mips(
    graph, patch_loader, gt_df, pred_df, output_dir, max_dist=32
):
    """
    Saves image and fragment-mask MIPs for every ground truth merge site.
    Filenames carry the highest prediction score within "max_dist", or
    "not_visited" if no prediction was scored nearby.

    Parameters
    ----------
    graph : FragmentsGraph
        Fragments graph the predictions were computed on.
    patch_loader : DetectionPatchLoader
        Loader that returns (node, patches) for a node in "graph".
    gt_df : pandas.DataFrame
        Ground truth sites with columns "unique_merge_id", "cell_id", "xyz".
    pred_df : pandas.DataFrame
        Predictions with columns "xyz" and "Prediction".
    output_dir : str
        Directory the PNGs are written to.
    max_dist : float, optional
        Radius (in microns) used to look up nearby predictions. Default is
        32.

    Returns
    -------
    List[dict]
        One record per site with the nearest graph node distance and the
        nearby prediction statistics.
    """
    os.makedirs(output_dir, exist_ok=True)
    if len(pred_df):
        pred_tree = KDTree(np.stack(pred_df["xyz"].values))
    else:
        pred_tree = None
    scores = pred_df["Prediction"].values

    records = []
    for _, site in gt_df.iterrows():
        xyz = np.asarray(site["xyz"], dtype=float)
        idxs = pred_tree.query_ball_point(xyz, max_dist) if pred_tree else []
        max_score = float(scores[idxs].max()) if idxs else float("nan")
        tag = f"score{max_score:.2f}" if idxs else "not_visited"

        node_dist, node = graph.kdtree.query(xyz)
        _, patches = patch_loader(node)
        output_path = os.path.join(
            output_dir, f"{site['unique_merge_id']}-{tag}.png"
        )
        img_util.plot_image_and_segmentation_mips(
            patches[0], patches[1], output_path=output_path
        )
        records.append(
            {
                "unique_merge_id": site["unique_merge_id"],
                "cell_id": site["cell_id"],
                "xyz": tuple(xyz.tolist()),
                "node_dist": round(float(node_dist), 1),
                "max_score": max_score,
                f"n_pred_within_{max_dist}": len(idxs),
            }
        )
    return records
