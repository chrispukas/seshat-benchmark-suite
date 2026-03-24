#!/usr/bin/env python3

import argparse
import os
import subprocess
import polars as pl

from typing import Dict, List, Optional, Any, Tuple

import llm_benchmark.utils.dataset as dataset
import llm_benchmark.utils.seshat_requests as seshat_requests

import llm_benchmark.config as config

from llm_benchmark.utils.llm_interface.query_core import LLMInterfaceModule
from llm_benchmark.utils.llm_interface.models.local.qwen import QwenInterfaceModule
import llm_benchmark.utils.llm_interface.evaluation_utils as eutils




def main():
    parser = argparse.ArgumentParser(description="Query Qwen Model")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen-7B-Chat", help="Name of the Qwen model to use")
    parser.add_argument("--save_path", type=str, default="/rds/general/user/cp824/home/neurips_llms/llm-benchmark/db/10_02_2026_1/run 1_qwen/", help="Path to save the generated questions")
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

    question_instance: QwenInterfaceModule = QwenInterfaceModule(
        model_name="Qwen/Qwen-7B-Chat",
        local = True,
        trust_remote_code=True
    )

    run_per_polity(
        save_path=args.save_path,
        polity="wf",
        ds=ds,
        question_instance=question_instance
    )

    

def run_per_polity(save_path: str, 
                   polity: str, 
                   ds: dataset.Dataset, 
                   question_instance: LLMInterfaceModule
                   ) -> None :
    wf_identifiers: List[str] = ds.get_identifiers_by_parent(polity)
    unhydrated_dir: str = os.path.join(save_path, polity)
    hydrated_dir: str = os.path.join(save_path, f"{polity}_hydrated")
    answer_dir: str = os.path.join(save_path, f"{polity}_answers")
    os.makedirs(hydrated_dir, exist_ok=True)
    os.makedirs(answer_dir, exist_ok=True)

    for identifier in wf_identifiers:
        print(f"Processing identifier: {identifier}")
        csv_questions_name: str = f"{identifier.replace('/', '_')}_questions.csv"
        csv_answers_name: str = f"{identifier.replace('/', '_')}_answers.csv"

        hydrated_df: pl.DataFrame = eutils.hydrate(ds, 
                                           questions_dir=os.path.join(unhydrated_dir, csv_questions_name),
                                           write_dir=os.path.join(hydrated_dir, csv_questions_name),
                                           link_to_dataset=True,)

        question_instance.respond_to_questions(
            DatasetModule = ds.dataset_modules[identifier],
            params = {"max_new_tokens": 512, "temperature": 0.7},
            output_path = os.path.join(answer_dir, csv_answers_name),
        )



if __name__ == "__main__":
    main()