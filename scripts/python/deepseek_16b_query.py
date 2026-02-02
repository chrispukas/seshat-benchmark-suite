#!/usr/bin/env python3

import argparse
import subprocess


from typing import Dict, List, Optional, Any, Tuple

import llm_benchmark.utils.dataset as dataset
import llm_benchmark.utils.seshat_requests as seshat_requests

import llm_benchmark.config as config

from llm_benchmark.utils.llm_interface.models.local.deepseek_18b import DeepSeekQuestionGenerationModule




def main():
    parser = argparse.ArgumentParser(description="Query DeepSeek 16B Model")
    parser.add_argument("--model_name", type=str, default="deepseek-moe-16b-base", help="Name of the DeepSeek model to use")
    args = parser.parse_args()


    polity_mapping: Dict[str, str] = config.polity_mapping

    module_dir: str = "/rds/general/user/cp824/home/neurips_llms/llm-benchmark/llm_benchmark/db/seshat/modules"
    main_dir: str = "/rds/general/user/cp824/home/neurips_llms/llm-benchmark/llm_benchmark/db/seshat"

    endpoint_identifiers: Dict[str, str] = seshat_requests.root_search_url(
        "https://seshat-db.com/api/", 
        use_cache=True, 
        cache_url="/rds/general/user/cp824/home/neurips_llms/llm-benchmark/cache/seshat_root_url.pkl"
    )

    print("Loaded endpoint identifiers:", endpoint_identifiers)

    ds: dataset.Dataset = dataset.Dataset(
        identifiers_endpoints=endpoint_identifiers,
        module_dir=module_dir,
        main_dir=main_dir,
        override=False,
        ignore_polities=["crisisdb/", "core/", "general/", "rt/"],
        polity_mapping=polity_mapping
    )

    print("Dataset initialized.")

    question_instance: DeepSeekQuestionGenerationModule = DeepSeekQuestionGenerationModule(
        model_name=args.model_name,
        local = True,
        trust_remote_code=True
    )

    print("DeepSeekQuestionGenerationModule initialized.")
    question_instance.generate_questions(
        dataset=ds.dataset_modules["wf/atlatls"],
        params={"max_new_tokens": 256, "temperature": 0.7},
        output_path="/rds/general/user/cp824/home/neurips_llms/llm-benchmark/db/29_01_2026_1/run 1_deepseek/deepseek_questions.csv"
    )

    print("Question generation completed.")


if __name__ == "__main__":
    main()