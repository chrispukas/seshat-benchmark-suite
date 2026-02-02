#!/bin/bash

#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1
#PBS -N deepseek_16b_query

LOCAL_DIR=/rds/general/user/cp824/home

module load Python/3.12.3-GCCcore-13.3.0

source "${LOCAL_DIR}/miniforge3/etc/profile.d/conda.sh"
conda activate llm

cd "${LOCAL_DIR}/neurips_llms/llm-benchmark/scripts/python/"
python "deepseek_16b_query.py"