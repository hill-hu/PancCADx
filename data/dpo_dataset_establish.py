"""
PancCADx - Error-Driven DPO Dataset Construction

Constructs preference pairs for Direct Preference Optimization (DPO)
by comparing SFT model predictions against ground truth labels.
Only incorrect predictions are used to form (chosen, rejected) pairs.

Usage:
    python dpo_dataset_establish.py \
        --sft_data_path data/sft_train.json \
        --predictions_path saves/sft_model/generated_predictions.jsonl \
        --output_path data/dpo_pairs.json
"""

import json
import os
import argparse


def build_dpo_dataset(sft_data_path: str, predictions_path: str, output_path: str):
    """
    Build DPO dataset from SFT training data and model predictions.
    
    The error-driven strategy:
    - Run inference on the training set using the SFT model
    - For samples where prediction != label (errors):
        - chosen = ground truth label
        - rejected = model's incorrect prediction
    - This teaches the model to prefer correct outputs over its own mistakes
    
    Args:
        sft_data_path: Path to original SFT training data (JSON)
        predictions_path: Path to SFT model predictions (JSONL)
        output_path: Path to save DPO dataset (JSON)
    """
    print(f"Loading SFT data: {sft_data_path}")
    with open(sft_data_path, 'r', encoding='utf-8') as f:
        sft_data = json.load(f)

    print(f"Loading predictions: {predictions_path}")
    pred_lines = []
    with open(predictions_path, 'r', encoding='utf-8') as f:
        pred_lines = f.readlines()

    if len(sft_data) != len(pred_lines):
        print(f"Warning: SFT data ({len(sft_data)}) != predictions ({len(pred_lines)})")
        print("Processing with minimum length.")

    dpo_dataset = []
    processed_count = min(len(sft_data), len(pred_lines))
    diff_count = 0

    print("Building DPO dataset...")

    for i in range(processed_count):
        try:
            pred_entry = json.loads(pred_lines[i])
        except json.JSONDecodeError:
            print(f"Line {i+1} JSON parse failed, skipping.")
            continue

        sft_entry = sft_data[i]
        prediction = pred_entry.get('predict', '').strip()
        label = pred_entry.get('label', '').strip()

        # Only include samples where model prediction is WRONG
        if prediction != label:
            # Extract user prompt from SFT data
            original_user_content = ""
            if 'messages' in sft_entry and len(sft_entry['messages']) > 0:
                original_user_content = sft_entry['messages'][0]['content']

            # Construct DPO preference pair
            dpo_item = {
                "messages": [
                    {"from": "user", "value": original_user_content}
                ],
                "chosen": {
                    "from": "gpt",
                    "value": label  # Ground truth = preferred
                },
                "rejected": {
                    "from": "gpt",
                    "value": prediction  # Model error = dispreferred
                },
                "images": sft_entry.get('images', [])
            }

            dpo_dataset.append(dpo_item)
            diff_count += 1

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dpo_dataset, f, ensure_ascii=False, indent=2)

    print(f"\nResults:")
    print(f"  Total samples processed: {processed_count}")
    print(f"  Error samples (DPO pairs): {diff_count}")
    print(f"  Error rate: {diff_count/processed_count*100:.1f}%")
    print(f"  DPO dataset saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build DPO dataset from SFT errors")
    parser.add_argument("--sft_data_path", type=str, required=True,
                        help="Path to SFT training data (JSON)")
    parser.add_argument("--predictions_path", type=str, required=True,
                        help="Path to SFT model predictions on training set (JSONL)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for DPO dataset (JSON)")
    args = parser.parse_args()

    build_dpo_dataset(args.sft_data_path, args.predictions_path, args.output_path)
