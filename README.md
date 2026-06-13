# PancCADx: A Multimodal Framework for Pancreatic Cancer Diagnosis

[![MICCAI 2026](https://img.shields.io/badge/MICCAI-2026-blue)](https://conferences.miccai.org/2026)

Official implementation of **"PancCADx: A Multimodal Framework for Pancreatic Cancer Diagnosis"**, accepted at MICCAI 2026.

## Overview

PancCADx is a multimodal reasoning framework for pancreatic cancer diagnosis that integrates EUS images with clinical EMR data through chain-of-thought (CoT) reasoning. Built on Qwen3-VL-8B-Thinking, our framework achieves interpretable diagnosis through a two-stage alignment pipeline: Supervised Fine-Tuning (SFT) followed by Direct Preference Optimization (DPO).

### Key Features
- **Conclusion-First CoT Protocol**: Anchors the diagnostic conclusion before backtracking through reasoning steps
- **Multimodal Fusion**: Combines EUS images with 30 clinical EMR variables
- **Error-Driven DPO**: Constructs preference pairs from model's own prediction errors
- **Multi-center Validation**: Validated on external cohorts from 3 hospitals

### Main Results
| Metric | SFT Only | SFT + DPO |
|--------|----------|-----------|
| Sensitivity (External) | 93.62% | 95.74% |
| Specificity (External) | 77.78% | 77.78% |
| Accuracy (External) | 88.48% | 89.53% |


## Installation

```bash
# Clone this repository
git clone https://github.com/hill-hu/PancCADx.git
cd PancCADx

# Install dependencies
pip install -r requirements.txt

# Install LLaMA-Factory (required for training and inference)
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ..
```

## Project Structure

```
PancCADx/
├── configs/                    # Training configurations
│   ├── sft_config.yaml         # SFT training config
│   └── dpo_config.yaml         # DPO training config
├── data/
│   ├── dataset_establish.py    # SFT dataset construction
│   ├── dpo_dataset_establish.py # DPO preference pair construction
│   └── dataset_info.json       # Dataset registration for LLaMA-Factory
├── evaluation/
│   ├── evaluate.py             # Single-center evaluation
│   └── evaluate_multicenter.py # Multi-center evaluation with per-site breakdown
├── inference/
│   └── vllm_infer.sh           # vLLM batch inference script
├── requirements.txt
└── README.md
```

## Data Preparation

### 1. Organize Raw Data

Prepare your clinical data in Excel format with the following columns:
- Patient demographics (age, gender, BMI)
- Clinical history (smoking, alcohol, symptoms, medications)
- Imaging features (CT, MRI, EUS findings)
- Diagnosis label (5-class: CA/NET/AIP/CP/SPT)

Organize EUS images by patient name under `CA/` and `OTHERS/` subdirectories.

### 2. Build SFT Dataset

```bash
python data/dataset_establish.py
```

This generates a JSON file in ShareGPT format with multimodal conversations pairing EUS images and clinical information.

### 3. Build DPO Dataset

After SFT training and inference on the training set:

```bash
python data/dpo_dataset_establish.py \
    --sft_data_path data/train.json \
    --predictions_path saves/sft_model/train_predictions.jsonl \
    --output_path data/dpo_pairs.json
```

This constructs preference pairs where:
- **Chosen**: Ground truth label (correct diagnosis)
- **Rejected**: Model's incorrect prediction from SFT stage

## Training

### Stage 1: Supervised Fine-Tuning (SFT)

```bash
llamafactory-cli train configs/sft_config.yaml
```

Key hyperparameters:
- Base model: Qwen3-VL-8B-Thinking
- LoRA: rank=128, alpha=256, target=all
- Batch size: 2, Gradient accumulation: 4
- Learning rate: 1e-4, Epochs: 20
- enable_thinking: true
- freeze_multi_modal_projector: true
- freeze_vision_tower: false

### Stage 2: Direct Preference Optimization (DPO)

```bash
llamafactory-cli train configs/dpo_config.yaml
```

Key hyperparameters:
- Adapter: loads SFT checkpoint
- Batch size: 1, Learning rate: 1e-7
- Epochs: 10, Beta: 0.3
- DeepSpeed ZeRO Stage 3
- freeze_vision_tower: true

## Inference

```bash
# Using vLLM for fast batch inference
python LLaMA-Factory/scripts/vllm_infer.py \
    --model_name_or_path /path/to/Qwen3-VL-8B-Thinking \
    --adapter_name_or_path /path/to/dpo_checkpoint \
    --dataset your_test_dataset \
    --dataset_dir data \
    --template qwen3_vl \
    --enable_thinking true \
    --cutoff_len 4096 \
    --max_new_tokens 1024 \
    --temperature 0.6 \
    --top_p 0.95 \
    --save_name results/predictions.jsonl
```

## Evaluation

```bash
# Single evaluation
python evaluation/evaluate.py --input results/predictions.jsonl

# Multi-center breakdown
python evaluation/evaluate_multicenter.py --input results/predictions.jsonl
```

## Model Weights

| Model | Description | Link |
|-------|-------------|------|
| PancCADx-DPO | SFT+DPO LoRA adapter (final) | [shan1984/PancCADx-DPO](https://huggingface.co/shan1984/PancCADx-DPO) |

Base model: [Qwen3-VL-8B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking)

## Citation

```bibtex
@inproceedings{hu2026panccadx,
  title={PancCADx: A Multimodal Framework for Pancreatic Cancer Diagnosis},
  author={Hu, Shan and Xiao, Changhong and Qin, Xianzheng and Mei, Bin and Cheng, Bin and Wang, Zhongyuan},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for the training framework
- [Qwen3-VL](https://github.com/QwenLM/Qwen2.5-VL) for the base vision-language model
