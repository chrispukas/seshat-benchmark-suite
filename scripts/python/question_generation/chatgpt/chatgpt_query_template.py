#!/usr/bin/env python3

import os
import argparse
import llm_benchmark.config as config

from dotenv import load_dotenv

from typing import Dict, List

from llm_benchmark.utils.llm_interface.models.remote.chatgpt import ChatGPTInterfaceModule
from llm_benchmark.utils.llm_interface.generation_utils import savepath_formatting
from llm_benchmark.utils.benchmark import generate

def query(
        model_name: str = "gpt-5.2-2025-12-11", 
        api_key: str = "", 
        polities_to_evaluate: List[str] = ["sc"]
        ):
    if api_key is None or api_key == "":
        load_dotenv()
        api_key = os.environ["OPENAI_API_KEY"]

    defaults: Dict[str, str] = {
        "unhydrated_question_save_path": "/Users/apple/Documents/github/neurips_llms/llm-bechmark/db/gen",
        "cache_dir": "/Users/apple/Documents/github/neurips_llms/llm-bechmark/db/seshat"
    }

    parser = argparse.ArgumentParser(description="Query OpenAI Model")
    parser.add_argument("--model_name", type=str, default=model_name, help="Idenfitifer of the OpenAI model to use")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the generated questions")
    parser.add_argument("--cache_dir", type=str, default=defaults["cache_dir"], help="Path to the cache directory")
    args = parser.parse_args()

    final_save_path: str = savepath_formatting(model_name, defaults.get("unhydrated_question_save_path")) if args.save_path is None else args.save_path
    os.makedirs(final_save_path, exist_ok=True)

    model_instance: ChatGPTInterfaceModule = ChatGPTInterfaceModule(
        model_name=args.model_name,
        api_key=api_key
    )

    generate(
        save_path=final_save_path,
        LLMInterfaceModule=model_instance,
        seshat_cache_dir=args.cache_dir,
        polity_mapping=config.polity_mapping,
        polities_to_evaluate=polities_to_evaluate
    )