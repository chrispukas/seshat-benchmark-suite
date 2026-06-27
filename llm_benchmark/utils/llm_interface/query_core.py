import os
import torch
import polars as pl
import math
import json

import logging

from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Callable, Set
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from llm_benchmark.config import \
    params_to_question_mapping, question_generation_template_mapping, \
    GENERATION_TEMPERATURE, GENERATION_MAXTOKENS_PER_PROMPT, \
    EVALUATION_TEMPERATURE, EVALUATION_MAXTOKENS_PER_PROMPT, BATCH_SIZE

from llm_benchmark.utils import utility as util
from llm_benchmark.utils.dataset import DatasetModule
from llm_benchmark.utils.enums import DatasetType, Tags, Quality, QuestionType
from llm_benchmark.utils.llm_interface.templates import generation_templates as gen_tmps 
from llm_benchmark.utils.llm_interface.templates import evaluation_templates as eval_tmps 
from llm_benchmark.utils.llm_interface import evaluation_utils as eval_utils 
from llm_benchmark.utils.llm_interface import generation_utils as gen_utils 


class LLMInterfaceModule():
    def __init__(self,
                 model_name: str = ""):
        self.model_name: str = model_name
        logging.info(f"Initialized {self.__class__.__name__}")
        pass
    
    def generate_questions(
            self, 
            DatasetModule: DatasetModule,
            allowed_question_type: QuestionType,

            logger: logging.Logger,

            params: Optional[Dict[str, Any]] = None,
            output_path: str = "",
            batch_size: Optional[int] = 32,
            ) -> None:

        try:
            endpoint: str = DatasetModule.get_endpoint()
            sub_dir: str = endpoint.split("/")[-2]
        except:
            logger.warning("Warning: Failed to parse endpoint.")
            return
        
        logger.info(f"\n\nGenerating questions for dataset at endpoint: {sub_dir}")

        dataset: pl.DataFrame = DatasetModule.get_entries()
        success: bool = self._query_generic(
            dataset=dataset,
            output_path=output_path,
            endpoint=endpoint,
            allowed_question_type=allowed_question_type,

            single_format_callable=self._format_single_input,
            params=params,

            batch_size=batch_size,
            sub_dir=sub_dir,

            logger=logger,
        )

    def respond_to_questions(
            self,
            DatasetModule: DatasetModule,
            params: Optional[Dict[str, Any]] = None,
            output_path: str = "",
            batch_size: int = 32,
            ) -> None:
        endpoint: str = DatasetModule.get_endpoint()
        dataset: pl.DataFrame = DatasetModule.get_questions()

        success: bool = self._query_generic(
            dataset=dataset,
            output_path=output_path,
            endpoint=endpoint,
            
            single_format_callable=self._format_single_response,
            params=params,

            batch_size=batch_size,
            sub_dir="",
        )

    def _query_generic(
            self,
            dataset: pl.DataFrame,
            output_path: str,
            endpoint: str,
            allowed_question_type: QuestionType,

            single_format_callable: Optional[Callable],

            logger: logging.Logger,

            params: Optional[Dict[str, Any]] = None,
            batch_size: Optional[int] = BATCH_SIZE,

            sub_dir: Optional[str] = "",
            
    ) -> Optional[bool]:
        if params is None:
            logger.warning("Warning: No parameters provided for question generation. Using default settings.")
            params = {}
        if output_path == None or output_path == "":
            return False
        if dataset is None:
            logger.warning(f"Warning: No questions linked to dataset {endpoint}.")
            return False
        if endpoint is None or endpoint == "":
            logger.warning("Warning: No endpoint specified.")
            return False

        output_dirname: str = os.path.dirname(output_path)
        if output_dirname:
            os.makedirs(output_dirname, exist_ok=True)

        output_buffer: List[Tuple[str, Optional[str], List[Dict[str, Any]], Dict[str, Any]]] = self._batch_query\
            (
                dataset=dataset,
                format_callable=single_format_callable,
                allowed_question_type=allowed_question_type,

                sub_dir=sub_dir,
                params=params,
                batch_size=batch_size,

                logger=logger,
            )

        question_outputs: List[Dict[str, Any]] = self._batch_format_outputs\
            (
                itms=output_buffer,
                endpoint=endpoint,
                format_callable=self._format_single_output,

                logger=logger
            )

        if not question_outputs:
            logger.warning(f"Failed to generate questions for endpoint: {endpoint}.")
            return False
        
        self.write_outputs\
            (
                raw=question_outputs, 
                output_path=output_path,
                logger=logger
            )
        return True
        
    def _batch_query(
            self,
            dataset: pl.DataFrame,
            format_callable: Callable,
            allowed_question_type: QuestionType,

            logger: logging.Logger,

            sub_dir: str = "",
            params: Dict[str, Any] = {},
            batch_size: int = 32,

    ) -> List[Tuple[str, Optional[str], List[Dict[str, Any]], Dict[str, Any]]]:
        height: int = dataset.height
        batch_count: int = height // batch_size

        output_buffer: List[Tuple[str, Optional[str], List[Dict[str, Any]], Dict[str, Any]]] = []
        tqdm_outs = util.TqdmToLogger(logger=logger)

        for idx in tqdm\
                (
                    range(batch_count+1), 
                    desc=f"Processing batch queries for {self.model_name}, batch size: {BATCH_SIZE}",
                    total=batch_count+1,
                    file=tqdm_outs,
                ):
            batch_slice: List[Dict[str, Any]] = dataset.slice(offset=idx * batch_size, length=batch_size).to_dicts()
            if batch_slice == []:
                continue

            input = [
                format_callable(
                    itm=itm,
                    params=params,
                    sub_dir=sub_dir,
                    logger=logger,
                    allowed_question_type=allowed_question_type
                ) for itm in batch_slice]

            raw_outs: List[Tuple[str, Optional[str], List[Dict[str, Any]], Dict[str, Any]]] = self.query_model\
                (
                    messages=input,
                    temperature = params.get("temperature", GENERATION_TEMPERATURE),
                    max_tokens = params.get("max_tokens", GENERATION_MAXTOKENS_PER_PROMPT),
                    logger=logger
                )
            
            output_buffer.extend(raw_outs)
        return output_buffer
    
    def _batch_format_outputs(
            self,
            itms: List[Tuple[str, Optional[str], List[Dict[str, Any]], Dict[str, Any]]],
            endpoint: str,
            format_callable: Callable,
            logger: logging.Logger,
            ) -> List[Dict[str, Any]]:
        question_outputs: List[Dict[str, Any]] = [{}] * len(itms)
        tqdm_outs = util.TqdmToLogger(logger=logger)

        total_token_sum: int = 0
        prompt_token_sum: int = 0

        for idx, (itm, reasoning, message, others) in tqdm\
            (
                enumerate(itms), 
                desc=f"Formatting items for endpoint: {endpoint}",
                total = len(itms),
                file = tqdm_outs,
            ):
            if itm is None:
                itm = ""

            question_outputs[idx] = format_callable(
                itm=itm,
                endpoint=endpoint,
                idx=idx,
                reasoning=reasoning,
                message=message,
                others=others,
            )

            if "usage.total_tokens" in others and "usage.prompt_tokens" in others:
                total_token_sum += others["usage.total_tokens"]
                prompt_token_sum += others["usage.prompt_tokens"]
        
        if total_token_sum > 0:
            logger.info(f"Total token usage: {total_token_sum}")
        if prompt_token_sum > 0:
            logger.info(f"Prompt token usage: {prompt_token_sum}")

        return question_outputs
        
    def _format_single_input(
            self,
            itm: Dict[str, Any],
            params: Dict[str, Any],
            sub_dir: str,
            logger: logging.Logger,
            allowed_question_type: QuestionType
        ) -> str:
        
        if not self.check_if_question_in_filter(itm, params, logger=logger):
            logger.warning(f"Warning: Skipping question ID {itm.get("id", "unknown")} due to filter settings.")
            return None
        
        raw_type: str = itm.get("question_type", "")
        question_types: Set[QuestionType] = util.question_types_remap(
            raw_type, 
            row=itm,
            endpoint=sub_dir.lower(),
            logger=logger
            )
        
        if allowed_question_type not in question_types:
            logger.warning(f"Skipping {allowed_question_type} due to filter settings.")
            return None

        template_method: Optional[Callable] = self._get_template(question_type=allowed_question_type)
        if not template_method:
            logger.warning(f"Warning: Template method for question type {allowed_question_type} not set in the configuration file.")
            return None
        
        try:
            remapped_item: Dict[str, Any] = gen_utils.remap_question_row(row=itm, endpoint=sub_dir)
            message_single: List[Dict[str, str]] = template_method(remapped_item)
        except Exception as e:
            logger.warning(f"Warning: failed to generate template for type '{allowed_question_type}'. Error: {e}")
            return None
        
        return message_single

    def _format_single_response(
            self,
            itm: Dict[str, Any],
            params: Dict[str, Any],
            sub_dir: str,  
    ) -> str:
        question: str = itm.get("output", "")
        if not question:
            return ""
        message_single: Dict[str, Any] = eval_tmps.multichoice(question)
        
        return message_single
    
    def _format_single_output(
            self,
            endpoint: str,
            idx: int,
            itm: str,
            reasoning: Optional[str],
            message: List[Dict[str, Any]],
            others: Dict[str, Any]
        ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "endpoint_identifier": endpoint, 
            "model": self.model_name,
            "entry_idx": idx,
            "output": itm,
            "reasoning": reasoning,
            "message": json.dumps(message) if isinstance(message, list) else message,
            "others": json.dumps(others) if isinstance(others, dict) else others,
            }
        return result
        
    

    def query_model(
            self, 
            messages: List[str],
            logger: logging.Logger,
            temperature: float = 0.7,
            max_tokens: int = 300,
            seed: int = 42,
            endpoint: str = "",
            ) -> List[Tuple[str, Optional[str], List[Dict[str, Any]], Dict[str, Any]]]:
        raise NotImplementedError("This method should be overridden by subclasses.")

    def _get_template(self, question_type: QuestionType) -> Callable:
        return question_generation_template_mapping.get(question_type, None)

    def _respond_batch(
            self, 
            row: Dict[str, Any], 
            params: Dict[str, Any],
            logger: logging.Logger
            ) -> str:
        question: str = row.get("output", "")
        if not question:
            return None
        prompt: Dict[str, Any] = eval_tmps.multichoice(question)
        return self.query_model(
            prompt,
            temperature=params.get("temperature", EVALUATION_TEMPERATURE),
            max_tokens=params.get("max_tokens", EVALUATION_MAXTOKENS_PER_PROMPT),
            logger=logger
            )

    def write_outputs(
            self, 
            raw: List[Dict[str, Any]], 
            output_path: str,
            logger: logging.Logger
            ) -> None:
        """Writes the generated questions to a CSV file."""
        if not output_path:
            logger.warning(f"Warning: Output path doesnt exist.")
            return
        if not raw:
            logger.warning(f"Warning: Failed to generate questions, no raw dicts found!")
            return
        if len(raw) == 0: 
            logger.warning(f"Warning: Empty raws, skipping this endpoint")
            return
        
        try:
            logger.info(raw[0])
            pl.DataFrame(raw).write_csv(output_path)
        except Exception as e:
            logger.warning(f"Warning: Error occurred while writing outputs to {output_path}: {e}")

        logger.info(f"Questions generated and saved to {output_path}")
    
    # Utility
    def hugging_face_model_load(
            self,
            model_name: str,
            trust_remote_code: Optional[bool] = False,
            local: Optional[bool] = True,
            cache_dir: Optional[str] = "/rds/general/user/cp824/ephemeral/huggingface_cache"
            ) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:        
        print(f"Loading model {model_name}.")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=trust_remote_code,
            local_files_only=local,
            cache_dir=cache_dir,
            )
        print("Tokenizer loaded successfully.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype="auto", 
            device_map="auto", 
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            local_files_only=local
            )
        try:
            model.generation_config = GenerationConfig.from_pretrained(
                model_name
                )
        except Exception as e:
            print(f"Failed to load generation config for {model_name}. Rollback to default config. Error: {e}")
            model.generation_config = GenerationConfig()
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

        return tokenizer, model
    
    def pull_remote_weights(
            self,
            model_name: str,
            trust_remote_code: Optional[bool] = False,
            cache_dir: Optional[str] = "/rds/general/user/cp824/ephemeral/huggingface_cache",
            ) -> None:
        self.hugging_face_model_load(model_name=model_name,
                                     trust_remote_code=trust_remote_code, 
                                     local=False,
                                     cache_dir=cache_dir,)
        
    def check_if_question_in_filter(
            self,
            question: Dict[str, Any],
            params: Dict[str, Any],
            logger: logging.Logger
            ) -> bool:
        """Check if a question matches the filter parameters."""
        for param_key, (question_key, remap) in params_to_question_mapping.items(): 
            param_item: List[Any] = params.get(param_key, [])
            question_item: Any = remap(question.get(question_key, None), question, logger=logger)

            if param_item and question_item not in param_item:
                return False
        return True