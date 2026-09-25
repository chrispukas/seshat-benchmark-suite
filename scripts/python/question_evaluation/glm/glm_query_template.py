#!/usr/bin/env python3

import os
import argparse

from typing import Dict

from llm_benchmark.utils.llm_interface.models.local.GLM import GLMInterfaceModule
from llm_benchmark.utils.benchmark import evaluate
from llm_benchmark.utils.llm_interface.generation_utils import savepath_formatting


def setup_model(model_name: str):
    return GLMInterfaceModule(
        model_name=model_name,
        local = False,
        trust_remote_code=True,
        pull_model=False,
        test_mode=True,
    )

def query(model_name: str = "THUDM/glm-4-9b"):
    defaults: Dict[str, str] = {
        "unhydrated_question_save_path": "/rds/general/user/cp824/home/neurips_llms/llm-benchmark/db/generation/",
        "hydrated_question_save_path": "/rds/general/user/cp824/home/neurips_llms/llm-benchmark/db/hydrated/",
        "answer_save_path": "/rds/general/user/cp824/home/neurips_llms/llm-benchmark/db/evaluation/"
    }

    parser = argparse.ArgumentParser(description="Query GLM Model")
    parser.add_argument("--model_name", type=str, default=model_name, help="Name of the GLM model to use")
    
    parser.add_argument("--unhydrated_question_save_path", type=str, default=defaults.get("unhydrated_question_save_path"), help="Path to save the generated questions")
    parser.add_argument("--hydrated_question_save_path", type=str, default=defaults.get("hydrated_question_save_path"), help="Path to save the generated questions")
    parser.add_argument("--answer_save_path", type=str, default=defaults.get("answer_save_path"), help="Path to save the generated answers")
    
    parser.add_argument("--cache_dir", type=str, default="/rds/general/user/cp824/home/neurips_llms/llm-benchmark/db/seshat", help="Path to the cache directory")
    args = parser.parse_args()

    final_answer_save_path: str = savepath_formatting(model_name, args.answer_save_path)
    os.makedirs(final_answer_save_path, exist_ok=True)

    question_instance: GLMInterfaceModule = setup_model(model_name=args.model_name)

    evaluate(
        unhydrated_question_save_path=args.unhydrated_question_save_path,
        answer_save_path=final_answer_save_path,
        llm_interface=question_instance,
        seshat_cache_dir=args.cache_dir,
        categories_to_evaluate=["wf"],
    )

if __name__ == "__main__":
    query()