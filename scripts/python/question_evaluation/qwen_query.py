#!/usr/bin/env python3

import argparse
import os
import subprocess

from typing import Dict, List, Optional, Any, Tuple
from llm_benchmark.utils.llm_interface.models.local.qwen import QwenInterfaceModule
from llm_benchmark.utils import evaluate


def main():
    parser = argparse.ArgumentParser(description="Query Qwen Model")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen-7B-Chat", help="Name of the Qwen model to use")
    parser.add_argument("--save_path", type=str, default="/rds/general/user/cp824/home/neurips_llms/llm-benchmark/db/10_02_2026_1/run 1_qwen/", help="Path to save the generated questions")
    parser.add_argument("--cache_dir", type=str, default="/rds/general/user/cp824/home/neurips_llms/llm-benchmark/llm_benchmark/db/seshat", help="Path to the cache directory")
    args = parser.parse_args()

    question_instance: QwenInterfaceModule = QwenInterfaceModule(
        model_name="Qwen/Qwen-7B-Chat",
        local = True,
        trust_remote_code=True
    )

    evaluate.evaluate(
        evaluation_save_path=args.save_path,
        LLMInterfaceModule=question_instance,
        seshat_cache_dir=args.cache_dir,
        polities_to_evaluate=["wf"],
    )

if __name__ == "__main__":
    main()