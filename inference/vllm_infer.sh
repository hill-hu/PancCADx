#!/bin/bash
# PancCADx - vLLM Batch Inference Script
# 
# This script uses LLaMA-Factory's vllm_infer.py for efficient batch inference.
# Requires: LLaMA-Factory installed, vLLM installed
#
# Usage: bash inference/vllm_infer.sh

# --- Configuration ---
MODEL_PATH="Qwen/Qwen3-VL-8B-Thinking"           # Base model path
ADAPTER_PATH="saves/panccadx_dpo"                  # LoRA adapter (DPO checkpoint)
DATASET="panccadx_test"                            # Dataset name in dataset_info.json
DATASET_DIR="data"                                 # Dataset directory
OUTPUT_DIR="results"                               # Output directory

# --- Inference Parameters ---
TEMPLATE="qwen3_vl"
CUTOFF_LEN=4096
MAX_NEW_TOKENS=1024
TEMPERATURE=0.6
TOP_P=0.95
BATCH_SIZE=512

# --- Create output directory ---
mkdir -p ${OUTPUT_DIR}

# --- Run Inference ---
python LLaMA-Factory/scripts/vllm_infer.py \
    --model_name_or_path ${MODEL_PATH} \
    --adapter_name_or_path ${ADAPTER_PATH} \
    --dataset ${DATASET} \
    --dataset_dir ${DATASET_DIR} \
    --template ${TEMPLATE} \
    --cutoff_len ${CUTOFF_LEN} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --enable_thinking true \
    --batch_size ${BATCH_SIZE} \
    --save_name ${OUTPUT_DIR}/generated_predictions.jsonl \
    --matrix_save_name ${OUTPUT_DIR}/metrics.json

echo "Inference complete. Results saved to ${OUTPUT_DIR}/"
