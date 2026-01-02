#!/usr/bin/env bash
export NOW=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="output/qwen2.5_1.5b_instruct_grpo_${NOW}"
CONFIG="recipes/Qwen2.5-Qwen-1.5B-Instrcut/ans_double.yaml"
mkdir -p "${OUT_DIR}"
export PYTHONPATH=$PYTHONPATH:~/zc/SandwichR

TRANSFORMERS_VERBOSITY=error \
CUDA_VISIBLE_DEVICES=0 \
python src/open_r1/rag_grpo_stage2.py \
    --output_dir ${OUT_DIR} \
    --config ${CONFIG}
