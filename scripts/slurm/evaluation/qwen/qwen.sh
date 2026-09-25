#!/bin/sh

#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1
#PBS -N qwen_7B

# LOCATING PYTHON VERS.
echo "Checking Python path:"
which python
python -c "import sys; print('\n'.join(sys.path))"

# SETTING HPC DIRS.
LOCAL_DIR=/rds/general/user/cp824/home
REPO_ROOT="${LOCAL_DIR}/neurips_llms/llm-benchmark"
SCRIPT_DIR="${REPO_ROOT}/scripts/python/question_evaluation/qwen"
PYTHON_SCRIPT="${SCRIPT_DIR}/qwen_query_template.py"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# INITIALIZING CONDA VENV
echo "Activating conda environment"
source "${LOCAL_DIR}/miniforge3/etc/profile.d/conda.sh"
conda activate qwen

cd "${SCRIPT_DIR}"

models=(
    "Qwen/Qwen-7B" 
    "Qwen/Qwen1.5-7B" 
    "Qwen/Qwen2-7B" 
    "Qwen/Qwen2.5-7B" 
    "Qwen/Qwen3-4B" 
    "Qwen/Qwen3-8B" 
    "Qwen/Qwen3.5-4B" 
    "Qwen/Qwen3.5-9B" 
    "Qwen/Qwen3.6-27B"
)
datasets=(
    "${REPO_ROOT}/db/gen/final/gpt-5.2-2025-12-11" 
    "${REPO_ROOT}/db/gen/final/gemini-2.5-flash" 
    )

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        echo "Processing dataset: $dataset, and model: $model."
        python "${PYTHON_SCRIPT}" \
            --model_name "${model}" \
            --unhydrated_question_save_path "${dataset}"
    done
done