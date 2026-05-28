#!/bin/sh

#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1
#PBS -N qwen3_8B

LOCAL_DIR=/rds/general/user/cp824/home
REPO_ROOT="${LOCAL_DIR}/neurips_llms/llm-benchmark"

echo "Activating conda environment"
source "${LOCAL_DIR}/miniforge3/etc/profile.d/conda.sh"
conda activate qwen

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

echo "Checking Python path:"
which python
python -c "import sys; print('\n'.join(sys.path))"

echo "Running Qwen query script"
cd "${REPO_ROOT}/scripts/python/question_generation/qwen"
python "qwen_Qwen3-8B.py"