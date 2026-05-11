#!/usr/bin/env python3

import os
import argparse

from llm_benchmark.utils.llm_interface.models.local.qwen import QwenInterfaceModule
from llm_benchmark.utils.benchmark import evaluate
from llm_benchmark.utils.llm_interface.generation_utils import savepath_formatting


def setup_model(model_name: str):
    return QwenInterfaceModule(
        model_name=model_name,
        local = True,
        trust_remote_code=True
    )

def query(model_name: str = "Qwen/Qwen-7B-Chat"):
    question_instance: QwenInterfaceModule = setup_model(model_name=model_name)
    return

    default_load_path: str = f"/rds/general/user/cp824/home/neurips_llms/llm-benchmark/db/generation/"
    default_save_path: str = f"/rds/general/user/cp824/home/neurips_llms/llm-benchmark/db/evaluation/"
    
    parser = argparse.ArgumentParser(description="Query Qwen Model")
    parser.add_argument("--model_name", type=str, default=model_name, help="Name of the Qwen model to use")
    parser.add_argument("--question_save_path", type=str, default=default_load_path, help="Path to save the generated questions")
    parser.add_argument("--answer_save_path", type=str, default=None, help="Path to save the generated answers")
    parser.add_argument("--cache_dir", type=str, default="/rds/general/user/cp824/home/neurips_llms/llm-benchmark/db/seshat", help="Path to the cache directory")
    args = parser.parse_args()

    final_answer_save_path: str = savepath_formatting(model_name, default_save_path) if args.answer_save_path is None else args.answer_save_path
    print(final_answer_save_path, args.answer_save_path)
    os.makedirs(final_answer_save_path, exist_ok=True)

    question_instance: QwenInterfaceModule = setup_model(model_name=args.model_name)

    evaluate(
        question_save_path=args.question_save_path,
        answer_save_path=final_answer_save_path,
        LLMInterfaceModule=question_instance,
        seshat_cache_dir=args.cache_dir,
        polities_to_evaluate=["wf"],
    )

if __name__ == "__main__":
    query()