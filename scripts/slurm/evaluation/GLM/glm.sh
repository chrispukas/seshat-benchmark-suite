
#!/bin/sh

#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1
#PBS -N OLMO

# LOCATING PYTHON VERS.
echo "Checking Python path:"
which python
python -c "import sys; print('\n'.join(sys.path))"

# SETTING HPC DIRS.
LOCAL_DIR=/rds/general/user/cp824/home
REPO_ROOT="${LOCAL_DIR}/neurips_llms/llm-benchmark"
SCRIPT_DIR="${REPO_ROOT}/scripts/python/question_evaluation/glm"
PYTHON_SCRIPT="${SCRIPT_DIR}/glm_query_template.py"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# INITIALIZING CONDA VENV
echo "Activating conda environment"
source "${LOCAL_DIR}/miniforge3/etc/profile.d/conda.sh"
conda activate glm

cd "${SCRIPT_DIR}"

models=(
    "THUDM/glm-4-9b"
    "THUDM/glm-4-9b-chat"
    "THUDM/glm-4-9b-chat-1m"
    "THUDM/glm-4v-9b"
    "THUDM/glm-4-voice-9b"
    "THUDM/chatglm3-6b-base"
    "THUDM/chatglm3-6b"
    "THUDM/chatglm2-6b"
    "THUDM/chatglm-6b"
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