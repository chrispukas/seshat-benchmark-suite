#!/usr/bin/env python3

import os
import argparse
import llm_benchmark.config as config

from dotenv import load_dotenv

from typing import Dict, List

from llm_benchmark import config as cfg
from llm_benchmark.utils.llm_interface.models.remote.chatgpt import ChatGPTInterfaceModule
from llm_benchmark.utils.llm_interface.generation_utils import savepath_formatting
from llm_benchmark.utils.benchmark import evaluate
from llm_benchmark.utils.enums import QuestionType, QuestionHydrationOptions


def query(
        model_name: str = "", 
        api_key: str = "", 
        allowed_question_types: List[QuestionType] = [QuestionType.MULTIPLE_CHOICE],
        evaluation_type: QuestionHydrationOptions = QuestionHydrationOptions.PRESENT_ABSENT_UNKNOWN,
        categories_to_evaluate=["sc", "wf", "ec", "rt"],
        batch_size: int = cfg.BATCH_SIZE,
        unhydrated_question_save_path="/Users/apple/Documents/github/neurips_llms/llm-bechmark/db/gen/final/gpt-4.1-2025-04-14/",
        hydrated_question_save_path="/Users/apple/Documents/github/neurips_llms/llm-bechmark/db/hyd/pres_abs_unk",
        manual_reasoning: bool = True,
        ):
    if api_key is None or api_key == "":
        load_dotenv()
        api_key = os.environ["OPENAI_API_KEY"]
    
    defaults: Dict[str, str] = {
        "unhydrated_question_save_path": "/Users/apple/Documents/github/neurips_llms/llm-bechmark/db/gen/final/gpt-4.1-2025-04-14/",
        "hydrated_question_save_path":   "/Users/apple/Documents/github/neurips_llms/llm-bechmark/db/hyd/",
        "answer_save_path":              "/Users/apple/Documents/github/neurips_llms/llm-bechmark/db/eval/",
        "cache_dir":                     "/Users/apple/Documents/github/neurips_llms/llm-bechmark/db/seshat"
    }
    if unhydrated_question_save_path is not None or unhydrated_question_save_path != "":
        defaults["unhydrated_question_save_path"] = unhydrated_question_save_path
    if hydrated_question_save_path is not None or hydrated_question_save_path != "":
        defaults["hydrated_question_save_path"] = hydrated_question_save_path
 
    parser = argparse.ArgumentParser(description="Query Qwen Model")
    parser.add_argument("--model_name", type=str, default=model_name, help="Name of the Qwen model to use")
    
    parser.add_argument("--unhydrated_question_save_path", type=str, default=defaults["unhydrated_question_save_path"], help="Path to unhydrated questions")
    parser.add_argument("--hydrated_question_save_path", type=str, default=defaults["hydrated_question_save_path"], help="Path to save hydrated questions")
    parser.add_argument("--answer_save_path", type=str, default=defaults["answer_save_path"], help="Path to save the generated answers")
    parser.add_argument("--batch_size", type=str, default=batch_size, help="Batch size")
    
    parser.add_argument("--cache_dir", type=str, default=defaults["cache_dir"], help="Path to the cache directory")
    args = parser.parse_args()

    if args.model_name is None or args.model_name == "":
        raise ValueError("OpenAI model name is not defined!")

    final_answer_save_path: str = savepath_formatting(args.model_name, args.answer_save_path)
    os.makedirs(final_answer_save_path, exist_ok=True)

    model_instance: ChatGPTInterfaceModule = ChatGPTInterfaceModule(
        model_name=args.model_name,
        api_key=api_key
    )

    evaluate(
        unhydrated_question_save_path=args.unhydrated_question_save_path,
        hydrated_question_save_path=args.hydrated_question_save_path,
        answer_save_path=final_answer_save_path,
        llm_interface=model_instance,
        seshat_cache_dir=args.cache_dir,

        allowed_question_types=allowed_question_types,
        categories_to_evaluate=categories_to_evaluate,
        evaluation_type=evaluation_type,

        manual_reasoning=manual_reasoning,

        batch_size=int(args.batch_size),
    )

if __name__ == "__main__":
    query()