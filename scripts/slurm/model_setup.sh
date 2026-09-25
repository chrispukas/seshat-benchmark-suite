#!/bin/bash
set -euo pipefail

export HF_HOME="/rds/general/user/cp824/home/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"

source "/rds/general/user/cp824/home/miniforge3/etc/profile.d/conda.sh"
conda activate qwen

models=(
    "deepseek-ai/deepseek-llm-7b-base"
    "deepseek-ai/deepseek-coder-6.7b-base"
    "deepseek-ai/deepseek-math-7b-base"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    "deepseek-ai/DeepSeek-V2-Lite-Base"

    "THUDM/glm-4-9b"
    "THUDM/glm-4-9b-chat"
    "THUDM/glm-4-9b-chat-1m"
    "THUDM/glm-4v-9b"
    "THUDM/glm-4-voice-9b"
    "THUDM/chatglm3-6b-base"
    "THUDM/chatglm3-6b"
    "THUDM/chatglm2-6b"
    "THUDM/chatglm-6b"

    "allenai/OLMo-1B-hf"
    "allenai/OLMo-7B-hf"
    "allenai/OLMo-2-1124-7B"
    "allenai/OLMo-2-1124-13B"
    "allenai/Olmo-3-1025-7B"
    "allenai/Olmo-3-7B-Instruct"
    "allenai/Olmo-3-1125-32B"
    "allenai/Olmo-3-32B-Think"
    "allenai/OLMoE-1B-7B-0924"

    "Qwen/Qwen-7B" 
    "Qwen/Qwen1.5-7B" 
    "Qwen/Qwen2-7B" 
    "Qwen/Qwen2.5-7B" 
    "Qwen/Qwen3-4B" 
    "Qwen/Qwen3-8B" 
    "Qwen/Qwen3.5-4B" 
    "Qwen/Qwen3.5-9B" 
    "Qwen/Qwen3.6-27B"
    "Qwen/Qwen3.8-27B"
)

for model in "${models[@]}"; do
    echo "Downloading ${model}"
    huggingface-cli download "${model}" \
        --cache-dir "${HF_HUB_CACHE}"
done