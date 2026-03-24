#!/usr/bin/env python3

import argparse
import llm_benchmark.config as config

from llm_benchmark.utils.llm_interface.models.remote.olmo import OlmoInterfaceModule
from llm_benchmark.utils.benchmark import generate

def main():
    parser = argparse.ArgumentParser(description="Query Olmo Model")
    parser.add_argument("--model_name", type=str, default="allenai/Olmo-3-1025-7B", help="Name of the Olmo model to use")
    parser.add_argument("--save_path", type=str, default="/rds/general/user/cp824/home/neurips_llms/llm-benchmark/db/02_03_2026_1/run 1_olmo/", help="Path to save the generated questions")
    parser.add_argument("--cache_dir", type=str, default="/rds/general/user/cp824/home/neurips_llms/llm-benchmark/llm_benchmark/db/seshat", help="Path to the cache directory")
    args = parser.parse_args()

    question_instance: OlmoInterfaceModule = OlmoInterfaceModule(
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