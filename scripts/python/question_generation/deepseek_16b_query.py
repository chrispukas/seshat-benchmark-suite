#!/usr/bin/env python3

import os
import argparse
import llm_benchmark.config as config

from llm_benchmark.utils.llm_interface.models.local.deepseek_18b import DeepSeekInterfaceModule
from llm_benchmark.utils.llm_interface.generation_utils import savepath_formatting
from llm_benchmark.utils.benchmark import generate

def main():
    model_name: str = "deepseek-ai/deepseek-moe-16b-base"
    default_save_path: str = f"/rds/general/user/cp824/home/neurips_llms/llm-benchmark/db/generation/"

    parser = argparse.ArgumentParser(description="Query DeepSeek Model")
    parser.add_argument("--model_name", type=str, default=model_name, help="Name of the DeepSeek model to use")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the generated questions")
    parser.add_argument("--cache_dir", type=str, default="/rds/general/user/cp824/home/neurips_llms/llm-benchmarks/db/seshat", help="Path to the cache directory")
    args = parser.parse_args()

    final_save_path: str = savepath_formatting(model_name, default_save_path) if args.save_path is None else args.save_path
    os.makedirs(final_save_path, exist_ok=True)

    question_instance: DeepSeekInterfaceModule = DeepSeekInterfaceModule(
        model_name=args.model_name,
        local = True,
        trust_remote_code=True
    )

    generate(
        save_path=args.save_path,
        LLMInterfaceModule=question_instance,
        seshat_cache_dir=args.cache_dir,
        polity_mapping=config.polity_mapping,
        polities_to_evaluate=["wf"]
    )
    

if __name__ == "__main__":
    main()