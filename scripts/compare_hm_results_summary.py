import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT)
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

# Reuse the dataset preparation and ranking evaluation setup.
report_path = ROOT / "reports" / "hm_ranking_evaluation.json"
if not report_path.exists():
    raise FileNotFoundError(f"Missing report: {report_path}")

ranking = json.loads(report_path.read_text(encoding="utf-8"))
training = json.loads(
    (ROOT / "reports" / "hm_sample_training_report.json").read_text(encoding="utf-8")
)

print("Existing two-tower ranking results:")
for key, value in ranking.get("metrics", {}).items():
    print(f"  {key}: {value:.4f}")

print("\nTraining report summary:")
for key in (
    "sampled_test_accuracy",
    "sampled_test_precision",
    "sampled_test_recall",
    "sampled_test_roc_auc",
):
    if key in training:
        print(f"  {key}: {training[key]}")

print(
    "\nNote: this script summarizes existing results only. "
    "It does not calculate a popularity baseline."
)
