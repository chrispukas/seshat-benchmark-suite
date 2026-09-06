#!/bin/bash

LOG_FILE="/Users/apple/Documents/github/neurips_llms/llm-bechmark/logs/eval_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gemini

SCRIPT_DIR="/Users/apple/Documents/github/neurips_llms/llm-bechmark/scripts/python/question_evaluation/gemini"
PYTHON_SCRIPT="${SCRIPT_DIR}/gemini_query_template.py"
#"gpt-3.5-turbo-0125" "gpt-3.5-turbo-1106" 
#"gemini-3.1-flash-lite" "gemini-2.5-flash"
#"/Users/apple/Documents/github/neurips_llms/llm-bechmark/db/gen/final/gpt-5.2-2025-12-11"
models=("gemini-2.5-pro")
datasets=("/Users/apple/Documents/github/neurips_llms/llm-bechmark/db/gen/final/gemini-2.5-flash" "/Users/apple/Documents/github/neurips_llms/llm-bechmark/db/gen/final/gemini-2.5-flash")

exec > >(tee -a "$LOG_FILE") 2>&1

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        echo "Processing: $model"
        python "$PYTHON_SCRIPT" --model_name "$model" --unhydrated_question_save_path "$dataset"
    done
done

