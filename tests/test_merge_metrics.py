"""Tests for merge detection metrics and the benchmark evaluator."""

import unittest

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from neuron_proofreading_evaluation.benchmarks import merge_detection
from neuron_proofreading_evaluation.metrics import merge_metrics


class MergeMetricsTest(unittest.TestCase):

    def setUp(self):
        self.gt_xyz = np.array(
            [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [200.0, 0.0, 0.0]]
        )
        # Two hits (one duplicated), one miss, one far false positive
        self.pred_xyz = np.array(
            [
                [5.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [100.0, 20.0, 0.0],
                [500.0, 0.0, 0.0],
            ]
        )

    def test_compute_metrics_counts_each_gt_once(self):
        result = merge_metrics.compute_metrics(
            KDTree(self.gt_xyz), self.pred_xyz, max_dist=25
        )
        self.assertEqual(result["# TP Sites"], 2)
        self.assertAlmostEqual(result["Recall"], 2 / 3, places=4)
        self.assertAlmostEqual(result["Precision"], 3 / 4, places=4)

    def test_compute_metrics_informedness(self):
        result = merge_metrics.compute_metrics(
            KDTree(self.gt_xyz), self.pred_xyz, max_dist=25, n_total=103
        )
        # 1 false positive among 100 negatives
        self.assertAlmostEqual(result["Informedness"], 2 / 3 - 0.01, places=3)

    def test_compute_metrics_without_predictions(self):
        result = merge_metrics.compute_metrics(
            KDTree(self.gt_xyz), np.empty((0, 3))
        )
        self.assertEqual(result["# Pred Sites"], 0)
        self.assertEqual(result["Recall"], 0)

    def test_filter_near_skeleton(self):
        pred_df = pd.DataFrame(
            {"xyz": list(map(tuple, self.pred_xyz)), "Prediction": 1.0}
        )
        skeleton = np.array([[x, 0.0, 0.0] for x in range(0, 201, 10)])
        kept = merge_metrics.filter_near_skeleton(
            pred_df, skeleton, max_dist=15
        )
        self.assertEqual(len(kept), 2)

    def test_reachable_recall(self):
        visited = np.array([[0.0, 10.0, 0.0], [100.0, 30.0, 0.0]])
        n_reachable, n_gt = merge_metrics.reachable_recall(
            self.gt_xyz, visited, max_dist=25
        )
        self.assertEqual((n_reachable, n_gt), (1, 3))

    def test_is_reachable_mask(self):
        visited = np.array([[0.0, 10.0, 0.0], [100.0, 30.0, 0.0]])
        mask = merge_metrics.is_reachable(self.gt_xyz, visited, max_dist=25)
        self.assertEqual(mask.dtype, bool)
        self.assertEqual((len(mask), int(mask.sum())), (3, 1))
        empty = merge_metrics.is_reachable(self.gt_xyz, np.empty((0, 3)))
        self.assertFalse(empty.any())

    def test_threshold_at_recall_interpolates(self):
        curve = pd.DataFrame(
            {"Threshold": [0.0, 0.5, 1.0], "Recall": [1.0, 0.9, 0.0]}
        )
        t = merge_metrics.threshold_at_recall(curve, target_recall=0.95)
        self.assertAlmostEqual(t, 0.25)

    def test_false_negatives(self):
        gt_df = pd.DataFrame(
            {
                "unique_merge_id": ["a", "b", "c"],
                "cell_id": "N",
                "xyz": list(map(tuple, self.gt_xyz)),
            }
        )
        pred_df = pd.DataFrame(
            {
                "xyz": list(map(tuple, self.pred_xyz)),
                "Prediction": [0.9, 0.2, 0.3, 0.9],
            }
        )
        fn_df = merge_metrics.false_negatives(gt_df, pred_df, 0.5)
        self.assertEqual(list(fn_df["unique_merge_id"]), ["b", "c"])
        self.assertAlmostEqual(fn_df["max_score"].iloc[0], 0.3)
        self.assertTrue(np.isnan(fn_df["max_score"].iloc[1]))


class MergeDetectionBenchmarkTest(unittest.TestCase):

    def test_parse_xyz_formats(self):
        for val in ["(1, 2, 3)", "[1 2 3]", (1, 2, 3), np.array([1, 2, 3])]:
            self.assertEqual(merge_detection.parse_xyz(val), (1.0, 2.0, 3.0))

    def test_evaluate_predictions_micro_and_macro(self):
        sol_df = pd.DataFrame(
            {
                "neuron_id": ["A", "A", "B"],
                "xyz_parsed": [
                    (0.0, 0.0, 0.0),
                    (100.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                ],
            }
        )
        sub_df = pd.DataFrame(
            {
                "neuron_id": ["A", "A", "B"],
                "xyz_parsed": [
                    (5.0, 0.0, 0.0),
                    (300.0, 0.0, 0.0),
                    (0.0, 50.0, 0.0),
                ],
            }
        )
        overall, per_neuron = merge_detection.evaluate_predictions(
            sub_df, sol_df, match_radius_um=25
        )
        self.assertEqual(overall["tp_gt"], 1)
        self.assertAlmostEqual(overall["recall"], 1 / 3, places=4)
        self.assertAlmostEqual(overall["precision"], 1 / 3, places=4)
        self.assertEqual(per_neuron["B"]["recall"], 0.0)
        self.assertAlmostEqual(overall["macro_recall"], 0.25, places=4)


if __name__ == "__main__":
    unittest.main()
