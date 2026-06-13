"""
PancCADx - Multi-Center Evaluation Script

Evaluates model performance with per-hospital breakdown for external validation.
Supports configurable data splits for different center cohorts.

Usage:
    python evaluate_multicenter.py --input results/external_predictions.jsonl
"""

import json
import re
import argparse
import pandas as pd


def extract_label(text: str) -> int:
    """
    Extract diagnosis from model output text.
    Removes <think>...</think> reasoning blocks before classification.
    """
    text_cleaned = re.sub(r'<think>.*?</think>', '', str(text), flags=re.DOTALL)

    if "非癌变" in text_cleaned:
        return 0
    elif "癌变" in text_cleaned or "胰腺癌" in text_cleaned:
        return 1
    return -1


def calc_metrics(preds: list, labels: list) -> list:
    """Calculate binary classification metrics."""
    TP = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    TN = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
    FP = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    FN = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)

    total = TP + TN + FP + FN
    if total == 0:
        return [0] * 12

    accuracy = (TP + TN) / total
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0

    return [accuracy, precision, recall, f1,
            TP, FP, TN, FN,
            recall, specificity, precision, npv]


# Define center splits (index ranges in the concatenated test set)
# Modify these according to your data ordering
CENTER_SPLITS = {
    "Hospital-B": (0, 103),      # indices 0-102
    "Hospital-C": (103, 168),    # indices 103-167
    "Hospital-D": (168, 191),    # indices 168-190
}


def evaluate_multicenter(file_path: str):
    """Evaluate with per-center breakdown."""
    data_splits = {"All (External)": {"preds": [], "labels": []}}
    for center in CENTER_SPLITS:
        data_splits[center] = {"preds": [], "labels": []}

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                pred_val = extract_label(data.get('predict', ''))
                label_val = extract_label(data.get('label', ''))

                if pred_val != -1 and label_val != -1:
                    data_splits["All (External)"]["preds"].append(pred_val)
                    data_splits["All (External)"]["labels"].append(label_val)

                    # Assign to center based on index
                    for center, (start, end) in CENTER_SPLITS.items():
                        if start <= i < end:
                            data_splits[center]["preds"].append(pred_val)
                            data_splits[center]["labels"].append(label_val)
                            break
            except json.JSONDecodeError:
                continue

    # Format results as table
    metric_names = [
        "Accuracy", "Precision", "Recall (Sensitivity)", "F1 Score",
        "TP", "FP", "TN", "FN",
        "Sensitivity", "Specificity", "PPV", "NPV"
    ]

    results = {"Metric": metric_names}
    for key in data_splits:
        metrics = calc_metrics(data_splits[key]["preds"], data_splits[key]["labels"])
        formatted = []
        for i, val in enumerate(metrics):
            if i in [4, 5, 6, 7]:
                formatted.append(int(val))
            else:
                formatted.append(f"{val:.4f}")
        results[key] = formatted

    df = pd.DataFrame(results)
    print(f"\n{'='*80}")
    print(f"Multi-Center Evaluation: {file_path}")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"{'='*80}\n")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-center evaluation")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to predictions JSONL file")
    args = parser.parse_args()
    evaluate_multicenter(args.input)
