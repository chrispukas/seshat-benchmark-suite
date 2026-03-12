#!/usr/bin/env python3

import argparse
import subprocess


from typing import Dict, List, Optional, Any, Tuple

import llm_benchmark.utils.dataset as dataset
import llm_benchmark.utils.seshat_requests as seshat_requests

import llm_benchmark.config as config

from llm_benchmark.utils.llm_interface.models.remote.olmo import OlmoInterfaceModule




def main():
    parser = argparse.ArgumentParser(description="Query Olmo Model")
    parser.add_argument("--model_name", type=str, default="allenai/Olmo-3-1025-7B", help="Name of the Olmo model to use")
    parser.add_argument("--save_path", type=str, default="/rds/general/user/cp824/home/neurips_llms/llm-benchmark/db/02_03_2026_1/run 1_olmo/", help="Path to save the generated questions")
    args = parser.parse_args()

    polity_mapping: Dict[str, str] = config.polity_mapping

    module_dir: str = "/rds/general/user/cp824/home/neurips_llms/llm-benchmark/llm_benchmark/db/seshat/modules"
    main_dir: str = "/rds/general/user/cp824/home/neurips_llms/llm-benchmark/llm_benchmark/db/seshat"

    endpoint_identifiers: Dict[str, str] = seshat_requests.root_search_url(
        "https://seshat-db.com/api/", 
        use_cache=True, 
        cache_url="/rds/general/user/cp824/home/neurips_llms/llm-benchmark/cache/seshat_root_url.pkl"
    )

    ds: dataset.Dataset = dataset.Dataset(
        identifiers_endpoints=endpoint_identifiers,
        module_dir=module_dir,
        main_dir=main_dir,
        override=False,
        ignore_polities=["crisisdb/", "core/", "general/", "rt/"],
        polity_mapping=polity_mapping
    )

    question_instance: OlmoInterfaceModule = OlmoInterfaceModule(
        model_name=args.model_name,
        local = True,
        trust_remote_code=True
    )

    wf_identifiers: List[str] = ds.get_identifiers_by_parent("wf")
    wf_identifiers = [wf_identifiers[0]]

    for identifier in wf_identifiers:
        print(f"Processing identifier: {identifier}")
        question_instance.generate_questions(
            DatasetModule = ds.dataset_modules[identifier],
            params = {"max_new_tokens": 512, "temperature": 0.7},
            output_path = f"{args.save_path}wf/{identifier.replace('/', '_')}_questions.csv",
        )


if __name__ == "__main__":
    main()