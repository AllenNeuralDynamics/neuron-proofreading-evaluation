# neuron-proofreading-evaluation

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![Python](https://img.shields.io/badge/python->=3.10-blue?logo=python)

## Merge Detection

`metrics.merge_metrics` scores merge site predictions against
ground truth sites: predictions are first restricted to within 15 um of the
ground truth skeleton (`filter_near_skeleton`), then matched within 25 um
(`compute_metrics`). `run_evaluation` writes the score histogram,
precision-recall-F1 curves, per-neuron results, a performance summary at the
decision threshold and at the thresholds reaching 97% and 95% recall, and the
TP/FP/FN sites. `reachable_recall` gives the recall ceiling of a search
schedule, and `visualization.save_gt_site_mips` saves image and mask MIPs at every
ground truth site.

### Whole-Brain Merge Detection Benchmark

`evaluate-merge-submission` evaluates a submission to the benchmark at
`gs://allen-nd-goog/anna.grim/benchmarks/merge_detection/wb_794495_gt5` with
the same matching rule as the benchmark's own evaluator:

```bash
evaluate-merge-submission --submission /path/to/submission_dir
evaluate-merge-submission --leaderboard
evaluate-merge-submission --upload_submission /path/to/submission_dir
```

A submission directory holds `predictions.csv` (`cell_id`, `xyz` as
`"(x, y, z)"` in microns, `score`, optional `flagged_merge`), `metadata.json`,
and `inference_script.py`.
