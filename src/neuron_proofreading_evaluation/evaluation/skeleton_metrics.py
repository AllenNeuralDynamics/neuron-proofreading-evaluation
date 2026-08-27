"""
Created on Mon Aug 26 09:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Computes skeleton-based metrics for each step of the automated proofreading pipeline.

"""

from segmentation_skeleton_metrics.evaluate import Evaluator
from segmentation_skeleton_metrics.utils import util

import numpy as np
import os
import pandas as pd

from neuron_proofreading_evaluation import datamodules as data_util


def compute(results_manager, gt_graphs, step_name, output_dir, ignore_kdtree=None):
    # Load fragments
    fragment_graphs = load_fragments(gt_graphs, results_manager.step_swcs_paths[step_name])

    # Create output directory
    step_output_dir = os.path.join(output_dir, step_name)
    util.mkdir(step_output_dir)

    # Call evaluator
    evaluator = Evaluator(step_output_dir, results_prefix=step_name)
    print_step_details(results_manager, evaluator, step_name, step_output_dir)
    evaluator(gt_graphs, fragment_graphs)

    # Filter ignored merge sites and update saved results
    if ignore_kdtree is not None:
        filter_merge_sites(evaluator, ignore_kdtree)
        update_saved_results(evaluator, gt_graphs)

    # Save results
    is_split = step_name != "original" and results_manager.step_types[step_name] == "split"
    color = "# COLOR 1.0 1.0 0.0" if is_split else "# COLOR 0.0 0.8 0.8"
    data_util.set_graph_color(fragment_graphs, color)
    evaluator.save_fragments(gt_graphs, fragment_graphs)
    evaluator.save_merge_results(gt_graphs, fragment_graphs)
    #evaluator.save_mips(gt_graphs, fragment_graphs)


def filter_merge_sites(evaluator, ignore_kdtree, max_dist=30):
    """
    Removes detected merge sites that fall within max_dist of any site in
    the ignore list.
    """
    merge_metric = evaluator.metrics["# Merges"]
    if len(merge_metric.merge_sites) == 0:
        return

    world_coords = np.array(list(merge_metric.merge_sites["World"]))
    dists, _ = ignore_kdtree.query(world_coords)
    keep_mask = dists > max_dist
    n_removed = (~keep_mask).sum()
    if n_removed > 0:
        print(f"   Filtered {n_removed} ignored merge site(s)")
    merge_metric.merge_sites = merge_metric.merge_sites[keep_mask]


def update_saved_results(evaluator, gt_graphs):
    """
    Recomputes # Merges and Merge Rate from the (already-filtered) merge_sites
    and overwrites results.csv and results_overview.txt on disk.
    """
    results_path = os.path.join(evaluator.output_dir, f"{evaluator.prefix}results.csv")
    results = pd.read_csv(results_path, index_col=0)

    merge_sites = evaluator.metrics["# Merges"].merge_sites
    has_sites = not merge_sites.empty and "GroundTruth_ID" in merge_sites.columns

    for name in results.index:
        results.loc[name, "# Merges"] = int((merge_sites["GroundTruth_ID"] == name).sum()) if has_sites else 0

    results["Merge Rate"] = evaluator.derived_metrics["Merge Rate"](gt_graphs, results)
    results.to_csv(results_path, index=True)

    overview_path = os.path.join(evaluator.output_dir, f"{evaluator.prefix}results_overview.txt")
    if os.path.exists(overview_path):
        os.remove(overview_path)
    evaluator.report_summary(results)


def load_fragments(gt_graphs, swcs_path):
    fragment_graphs = data_util.load_fragments(swcs_path, use_anisotropy=True)
    data_util.relabel_fragments_with_name(fragment_graphs)
    data_util.relabel_groundtruth_wrt_fragments(gt_graphs, fragment_graphs)
    return fragment_graphs


def print_step_details(results_manager, evaluator, step_name, step_output_dir):
    log_path = os.path.join(step_output_dir, f"{evaluator.prefix}results_overview.txt")
    util.mkdir(step_output_dir)
    if os.path.exists(log_path):
        os.remove(log_path)

    util.update_txt(log_path, "\nExperiment Details")
    util.update_txt(log_path, "-" * (len(results_manager.s3_prefix) + 9))
    util.update_txt(log_path, f"Brain_ID: {results_manager.brain_id}")
    util.update_txt(log_path, f"Segmentation_ID: {results_manager.segmentation_id}")
    util.update_txt(log_path, f"S3 Prefix: {results_manager.s3_prefix}")
    util.update_txt(log_path, f"Evaluation Step: {step_name}")
