"""
PancCADx - Evaluation Script

Evaluates binary classification performance (malignant vs. benign)
from model prediction files in JSONL format.

Usage:
    python evaluate.py --input results/predictions.jsonl
"""

import json
import re
import argparse
import pandas as pd


def extract_label(text: str) -> int:
    """
    Extract diagnosis from model output text.
    Removes <think>...</think> reasoning blocks before classification.
    
    Returns:
        0: Benign (非癌变)
        1: Malignant (癌变)
        -1: Unable to parse
    """
    # Remove chain-of-thought reasoning blocks
    text_cleaned = re.sub(r'<think>.*?</think>', '', str(text), flags=re.DOTALL)

    if "非癌变" in text_cleaned:
        return 0  # Benign
    elif "癌变" in text_cleaned or "胰腺癌" in text_cleaned:
        return 1  # Malignant
    return -1  # Unknown


def calc_all_metrics(preds: list, labels: list) -> dict:
    """Calculate comprehensive binary classification metrics."""
    TP = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    TN = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
    FP = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    FN = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)

    total = TP + TN + FP + FN
    if total == 0:
        return {}

    accuracy = (TP + TN) / total
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0  # Sensitivity
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0

    return {
        "Accuracy": accuracy,
        "Precision (PPV)": precision,
        "Sensitivity (Recall)": recall,
        "Specificity": specificity,
        "F1 Score": f1,
        "NPV": npv,
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "Total": total
    }


def evaluate(file_path: str) -> dict:
    """Evaluate predictions from a JSONL file."""
    predictions = []
    labels = []
    parse_failures = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                pred_val = extract_label(data.get('predict', ''))
                label_val = extract_label(data.get('label', ''))

                if pred_val != -1 and label_val != -1:
                    predictions.append(pred_val)
                    labels.append(label_val)
                else:
                    parse_failures += 1
            except json.JSONDecodeError:
                parse_failures += 1

    metrics = calc_all_metrics(predictions, labels)

    print(f"\n{'='*60}")
    print(f"Evaluation Results: {file_path}")
    print(f"{'='*60}")
    print(f"Valid samples: {len(predictions)}, Parse failures: {parse_failures}")
    print(f"{'─'*60}")

    for key, value in metrics.items():
        if key in ["TP", "FP", "TN", "FN", "Total"]:
            print(f"  {key:25s}: {int(value)}")
        else:
            print(f"  {key:25s}: {value:.4f} ({value*100:.2f}%)")

    print(f"{'='*60}\n")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PancCADx predictions")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to predictions JSONL file")
    args = parser.parse_args()
    evaluate(args.input)
