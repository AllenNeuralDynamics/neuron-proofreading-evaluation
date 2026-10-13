"""
Created on Mon Oct 12 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Evaluation tool for the Whole-Brain Merge Detection Benchmark (wb_794495_gt5).

Portable counterpart of the evaluate_submission.py shipped in the benchmark
root, which only runs inside google3. It reproduces the same matching rule
and metrics so scores are directly comparable, and reads local or gs://
paths through neuron_proofreader.

    # Evaluate a submission directory (local or gs://) or a predictions.csv
    evaluate-merge-submission --submission /results/my_submission
    evaluate-merge-submission --submission gs://.../submissions/my_submission

    # Leaderboard across every submission directory
    evaluate-merge-submission --leaderboard

    # Upload a local submission directory to the benchmark's submissions/
    evaluate-merge-submission --upload_submission /results/my_submission

Matching rule: per neuron, a prediction matches a GT merge site if their
Euclidean distance is <= match_radius_um. Recall = GT sites with a prediction
within radius / total GT; precision = predictions within radius of a GT site
/ total predictions. Macro metrics average per-neuron values.

Submission layout: predictions.csv (cell_id, xyz "(x, y, z)" in microns,
score, optional flagged_merge), metadata.json, inference_script.py.

"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from io import StringIO

import numpy as np
import pandas as pd
from neuron_proofreader.utils import util
from scipy.spatial import KDTree

MATCH_RADIUS_UM = 25.0

DEFAULT_BENCHMARK_ROOT = (
    "gs://allen-nd-goog/anna.grim/benchmarks/merge_detection/wb_794495_gt5"
)
DEFAULT_SOLUTION_PATH = f"{DEFAULT_BENCHMARK_ROOT}/solution.csv"
DEFAULT_SUBMISSIONS_DIR = f"{DEFAULT_BENCHMARK_ROOT}/submissions"
DEFAULT_GT_SWCS_DIR = f"{DEFAULT_BENCHMARK_ROOT}/data/gt_swcs"
DEFAULT_PRED_SWCS_DIR = f"{DEFAULT_BENCHMARK_ROOT}/data/pred_swcs"

REQUIRED_METADATA_FIELDS = (
    "submission_name",
    "author",
    "method_description",
    "code_repository",
    "reproduction_command",
)
SUBMISSION_FILES = ("predictions.csv", "metadata.json", "inference_script.py")


# --- Loading ---
def parse_xyz(val):
    """
    Parses a 3D coordinate from a "(x, y, z)" string, tuple, list, or array.
    """
    if isinstance(val, (list, tuple, np.ndarray)):
        return (float(val[0]), float(val[1]), float(val[2]))
    if isinstance(val, str):
        val_clean = val.strip().strip("()[]")
        if "," in val_clean:
            parts = [p.strip() for p in val_clean.split(",")]
        else:
            parts = val_clean.split()
        parts = [p for p in parts if p]
        if len(parts) >= 3:
            return (float(parts[0]), float(parts[1]), float(parts[2]))
    raise ValueError(f"Cannot parse 3D coordinate from: {val!r}")


def read_csv(path):
    return pd.read_csv(StringIO(util.read_txt(path)))


def load_solution(solution_path=DEFAULT_SOLUTION_PATH):
    df = read_csv(solution_path)
    if "cell_id" in df.columns and "neuron_id" not in df.columns:
        df = df.rename(columns={"cell_id": "neuron_id"})
    df["xyz_parsed"] = [parse_xyz(v) for v in df["xyz"]]
    return df


def load_submission(submission_path, score_threshold=None):
    """
    Loads a submission CSV or directory. Returns (predictions, metadata),
    where predictions are filtered to active merge detections: score >=
    score_threshold if given, else the flagged_merge column.
    """
    sub_dir = None
    csv_path = submission_path
    if not submission_path.endswith(".csv"):
        sub_dir = submission_path.rstrip("/")
        csv_path = os.path.join(sub_dir, "predictions.csv")
    df = read_csv(csv_path)

    if "cell_id" in df.columns and "neuron_id" not in df.columns:
        df = df.rename(columns={"cell_id": "neuron_id"})

    if "xyz" in df.columns:
        df["xyz_parsed"] = [parse_xyz(v) for v in df["xyz"]]
    elif {"x", "y", "z"}.issubset(df.columns):
        df["xyz_parsed"] = list(
            zip(
                df["x"].astype(float),
                df["y"].astype(float),
                df["z"].astype(float),
            )
        )
    elif "xyz_um" in df.columns:
        df["xyz_parsed"] = [parse_xyz(v) for v in df["xyz_um"]]
    else:
        raise ValueError(
            "Submission must have 'xyz', 'xyz_um', or 'x','y','z' columns!"
        )

    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(1.0)
    elif "p_merge" in df.columns:
        df["score"] = pd.to_numeric(df["p_merge"], errors="coerce").fillna(1.0)
    elif "confidence" in df.columns:
        conf_map = {"high": 1.0, "medium": 0.8, "low": 0.6}
        df["score"] = [
            conf_map.get(str(c).lower(), 0.5) for c in df["confidence"]
        ]
    else:
        df["score"] = 1.0

    # Filter predictions to active merge error candidates
    if score_threshold is not None:
        df = df[df["score"] >= score_threshold]
    elif "flagged_merge" in df.columns:
        df = df[
            df["flagged_merge"].astype(str).str.lower().isin(["true", "1"])
        ]
    elif "kept_after_nms" in df.columns:
        df = df[
            df["kept_after_nms"].astype(str).str.lower().isin(["true", "1"])
        ]
    elif "decision" in df.columns:
        df = df[df["decision"] == "merge_error"]
    df = df.copy().reset_index(drop=True)

    metadata = None
    if sub_dir is not None:
        try:
            metadata = json.loads(
                util.read_txt(os.path.join(sub_dir, "metadata.json"))
            )
        except Exception:
            pass
    return df, metadata


# --- Metrics ---
def evaluate_single_neuron(pred_xyz, gt_xyz, match_radius_um=MATCH_RADIUS_UM):
    pred_arr = np.asarray(pred_xyz, dtype=np.float64).reshape(-1, 3)
    gt_arr = np.asarray(gt_xyz, dtype=np.float64).reshape(-1, 3)
    n_pred, n_gt = len(pred_arr), len(gt_arr)
    if n_gt == 0 or n_pred == 0:
        return {
            "n_gt": n_gt,
            "n_pred": n_pred,
            "tp_gt": 0,
            "fn_gt": n_gt,
            "tp_pred": 0,
            "fp_pred": n_pred,
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
        }

    gt_to_pred_dists, _ = KDTree(pred_arr).query(gt_arr)
    pred_to_gt_dists, _ = KDTree(gt_arr).query(pred_arr)
    tp_gt = int(np.sum(gt_to_pred_dists <= match_radius_um))
    tp_pred = int(np.sum(pred_to_gt_dists <= match_radius_um))
    recall = tp_gt / n_gt
    precision = tp_pred / n_pred
    f1 = _f1(precision, recall)
    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "tp_gt": tp_gt,
        "fn_gt": n_gt - tp_gt,
        "tp_pred": tp_pred,
        "fp_pred": n_pred - tp_pred,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
    }


def evaluate_predictions(sub_df, sol_df, match_radius_um=MATCH_RADIUS_UM):
    """
    Scores an already-loaded, already-filtered predictions frame against the
    solution frame. Returns (overall, per_neuron).
    """
    per_neuron = {}
    tot_tp_gt = tot_tp_pred = 0
    for nid in sol_df["neuron_id"].unique():
        gt_coords = list(sol_df.loc[sol_df["neuron_id"] == nid, "xyz_parsed"])
        if len(sub_df):
            pred_coords = list(
                sub_df.loc[sub_df["neuron_id"] == nid, "xyz_parsed"]
            )
        else:
            pred_coords = []
        res = evaluate_single_neuron(pred_coords, gt_coords, match_radius_um)
        per_neuron[nid] = res
        tot_tp_gt += res["tp_gt"]
        tot_tp_pred += res["tp_pred"]

    tot_gt, tot_pred = len(sol_df), len(sub_df)
    recall = tot_tp_gt / tot_gt if tot_gt else 0.0
    precision = tot_tp_pred / tot_pred if tot_pred else 0.0
    overall = {
        "n_gt": tot_gt,
        "n_pred": tot_pred,
        "tp_gt": tot_tp_gt,
        "tp_pred": tot_tp_pred,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(_f1(precision, recall), 4),
        "macro_recall": _macro(per_neuron, "recall"),
        "macro_precision": _macro(per_neuron, "precision"),
        "macro_f1": _macro(per_neuron, "f1"),
    }
    return overall, per_neuron


def evaluate_submission(
    submission_path,
    solution_path=DEFAULT_SOLUTION_PATH,
    match_radius_um=MATCH_RADIUS_UM,
    score_threshold=None,
):
    sol_df = load_solution(solution_path)
    sub_df, sub_meta = load_submission(submission_path, score_threshold)
    overall, per_neuron = evaluate_predictions(sub_df, sol_df, match_radius_um)

    if sub_meta is not None:
        missing = [f for f in REQUIRED_METADATA_FIELDS if not sub_meta.get(f)]
    else:
        missing = list(REQUIRED_METADATA_FIELDS)

    return {
        "submission_path": submission_path,
        "solution_path": solution_path,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "match_radius_um": match_radius_um,
        "score_threshold": score_threshold,
        "metadata": sub_meta or {},
        "reproducibility_check": {
            "has_metadata_file": sub_meta is not None,
            "is_fully_specified": len(missing) == 0,
            "missing_fields": missing,
        },
        "overall": overall,
        "per_neuron": per_neuron,
    }


def _f1(precision, recall):
    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0


def _macro(per_neuron, key):
    return round(float(np.mean([r[key] for r in per_neuron.values()])), 4)


# --- Reporting ---
def print_evaluation_report(report):
    print("\n" + "=" * 80)
    print("MERGE DETECTION BENCHMARK EVALUATION REPORT")
    print("=" * 80)
    print(f"Submission:  {report['submission_path']}")
    print(f"Solution:    {report['solution_path']}")
    print(f"Match Radius: {report['match_radius_um']} um")
    if report.get("score_threshold") is not None:
        print(f"Score Threshold: >= {report['score_threshold']}")

    meta = report.get("metadata", {})
    if meta:
        print("\n--- Submission Metadata ---")
        print(f"  Submission Name: {meta.get('submission_name', 'N/A')}")
        print(f"  Author:          {meta.get('author', 'N/A')}")
        print(f"  Model / Method:  {meta.get('model_name', 'N/A')}")
        print(f"  Description:     {meta.get('method_description', 'N/A')}")
        repo = meta.get("code_repository", {})
        if isinstance(repo, dict):
            print(
                f"  Repository:      {repo.get('repo_path', 'N/A')} "
                f"@ {repo.get('revision', 'N/A')}"
            )
        else:
            print(f"  Repository:      {repo}")
        print(f"  Run Command:     {meta.get('reproduction_command', 'N/A')}")

    repro = report.get("reproducibility_check", {})
    if repro.get("is_fully_specified"):
        print(
            "\n[OK] Reproducibility Check Passed "
            "(all required metadata specified)."
        )
    else:
        print(
            "\n[WARNING] Incomplete reproducibility metadata! "
            f"Missing fields: {repro.get('missing_fields')}"
        )

    print("\n--- Per-Neuron Performance ---")
    rows = [
        {
            "Neuron ID": nid,
            "GT Merges": r["n_gt"],
            "Pred Merges": r["n_pred"],
            "TP (GT)": r["tp_gt"],
            "TP (Pred)": r["tp_pred"],
            "Recall": f"{r['recall'] * 100:.2f}%",
            "Precision": f"{r['precision'] * 100:.2f}%",
            "F1 Score": f"{r['f1']:.4f}",
        }
        for nid, r in report["per_neuron"].items()
    ]
    print(pd.DataFrame(rows).to_string(index=False))

    ov = report["overall"]
    print("\n--- Overall Benchmark Performance ---")
    print(f"  Total Ground Truth Merges: {ov['n_gt']}")
    print(f"  Total Predicted Merges:    {ov['n_pred']}")
    print(f"  True Positives (GT):       {ov['tp_gt']} / {ov['n_gt']}")
    print(f"  True Positives (Pred):     {ov['tp_pred']} / {ov['n_pred']}")
    print(f"  Micro Recall:              {ov['recall'] * 100:.2f}%")
    print(f"  Micro Precision:           {ov['precision'] * 100:.2f}%")
    print(f"  Micro F1 Score:            {ov['f1']:.4f}")
    print(f"  Macro F1 Score:            {ov['macro_f1']:.4f}")
    print("=" * 80 + "\n")


def list_submission_dirs(submissions_dir):
    if util.is_gcs_path(submissions_dir):
        bucket, _ = util.parse_cloud_path(submissions_dir)
        return [
            f"gs://{bucket}/{p.rstrip('/')}"
            for p in util.list_gcs_subprefixes(submissions_dir)
        ]
    return util.list_subdirs(submissions_dir, return_paths=True)


def build_leaderboard(
    submissions_dir=DEFAULT_SUBMISSIONS_DIR,
    solution_path=DEFAULT_SOLUTION_PATH,
    match_radius_um=MATCH_RADIUS_UM,
):
    rows = []
    for s_dir in sorted(list_submission_dirs(submissions_dir)):
        s_name = s_dir.rstrip("/").split("/")[-1]
        csv_path = os.path.join(s_dir, "predictions.csv")
        if util.is_gcs_path(csv_path):
            exists = util.check_gcs_file_exists(csv_path)
        else:
            exists = os.path.exists(csv_path)
        if not exists:
            continue
        try:
            rep = evaluate_submission(s_dir, solution_path, match_radius_um)
        except Exception as e:
            print(
                f"Failed to evaluate submission at {s_dir}: {e}",
                file=sys.stderr,
            )
            continue
        ov, meta = rep["overall"], rep.get("metadata", {})
        rows.append(
            {
                "Submission": s_name,
                "Model / Method": meta.get("model_name", s_name),
                "Author": meta.get("author", "N/A"),
                "F1 Score": ov["f1"],
                "Recall": ov["recall"],
                "Precision": ov["precision"],
                "N Pred": ov["n_pred"],
                "Reproducible": rep["reproducibility_check"][
                    "is_fully_specified"
                ],
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("F1 Score", ascending=False)
    df = df.reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def upload_submission(local_dir, submissions_dir=DEFAULT_SUBMISSIONS_DIR):
    """
    Copies predictions.csv, metadata.json, and inference_script.py to the
    benchmark's submissions directory on GCS.
    """
    from google.cloud import storage

    name = os.path.basename(local_dir.rstrip("/"))
    bucket_name, prefix = util.parse_cloud_path(submissions_dir.rstrip("/"))
    bucket = storage.Client().bucket(bucket_name)
    for filename in SUBMISSION_FILES:
        src = os.path.join(local_dir, filename)
        if not os.path.exists(src):
            print(f"  skipping missing {filename}")
            continue
        dst = f"{prefix}/{name}/{filename}"
        bucket.blob(dst).upload_from_filename(src)
        print(f"  uploaded gs://{bucket_name}/{dst}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--submission",
        help="Submission CSV or directory with predictions.csv "
        "(local or gs://)",
    )
    parser.add_argument("--solution", default=DEFAULT_SOLUTION_PATH)
    parser.add_argument(
        "--leaderboard",
        action="store_true",
        help="Evaluate every directory in --submissions_dir",
    )
    parser.add_argument("--submissions_dir", default=DEFAULT_SUBMISSIONS_DIR)
    parser.add_argument(
        "--match_radius_um", type=float, default=MATCH_RADIUS_UM
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=None,
        help="Keep predictions with score >= this instead of the "
        "flagged_merge column",
    )
    parser.add_argument("--output_report", help="Write the JSON report here")
    parser.add_argument(
        "--upload_submission",
        metavar="LOCAL_DIR",
        help="Upload a local submission directory to --submissions_dir "
        "and exit",
    )
    args = parser.parse_args()

    if args.upload_submission:
        print(
            f"Uploading {args.upload_submission} to {args.submissions_dir} ..."
        )
        upload_submission(args.upload_submission, args.submissions_dir)
        return

    if args.leaderboard:
        print(
            "Generating leaderboard across submissions in "
            f"{args.submissions_dir} ..."
        )
        board = build_leaderboard(
            args.submissions_dir, args.solution, args.match_radius_um
        )
        if len(board) == 0:
            print("No evaluated submissions found in directory.")
        else:
            print("\n" + "=" * 80)
            print("MERGE DETECTION BENCHMARK LEADERBOARD")
            print("=" * 80)
            print(board.to_string(index=False))
            print("=" * 80 + "\n")
        return

    if not args.submission:
        parser.error(
            "Must specify --submission, --leaderboard, or --upload_submission."
        )

    report = evaluate_submission(
        args.submission,
        args.solution,
        args.match_radius_um,
        args.score_threshold,
    )
    print_evaluation_report(report)
    if args.output_report:
        util.write_json(args.output_report, report)
        print(f"Saved report to {args.output_report}")


if __name__ == "__main__":
    main()
