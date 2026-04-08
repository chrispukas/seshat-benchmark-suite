#!/usr/bin/env python3

import os
import argparse

from llm_benchmark.utils.llm_interface.models.remote.olmo import OlmoInterfaceModule
from llm_benchmark.utils.benchmark import evaluate
from llm_benchmark.utils.llm_interface.generation_utils import savepath_formatting


def main():
    model_name: str = "Qwen/Qwen-7B-Chat"
    default_save_path: str = f"/rds/general/user/cp824/home/neurips_llms/llm-benchmark/db/evaluation/"

    parser = argparse.ArgumentParser(description="Query Olmo Model")
    parser.add_argument("--model_name", type=str, default=model_name, help="Name of the Olmo model to use")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the generated questions")
    parser.add_argument("--cache_dir", type=str, default="/rds/general/user/cp824/home/neurips_llms/llm-benchmark/llm_benchmark/db/seshat", help="Path to the cache directory")
    args = parser.parse_args()

    final_save_path: str = savepath_formatting(model_name, default_save_path) if args.save_path is None else args.save_path
    os.makedirs(final_save_path, exist_ok=True)

    question_instance: OlmoInterfaceModule = OlmoInterfaceModule(
        model_name=args.model_name,
        local = True,
        trust_remote_code=True
    )

    evaluate(
        evaluation_save_path=args.save_path,
        LLMInterfaceModule=question_instance,
        seshat_cache_dir=args.cache_dir,
        polities_to_evaluate=["wf"],
    )

if __name__ == "__main__":
    main()