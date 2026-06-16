import os
import torch
import polars as pl
import math

from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Callable
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from llm_benchmark.config import params_to_question_mapping, question_generation_template_mapping, GENERATE_TEMPERATURE, GENERATE_TOKENS_PER_PROMPT, QUERY_TEMPERATURE, QUERY_TOKENS_PER_PROMPT

from llm_benchmark.utils import utility as util
from llm_benchmark.utils.dataset import DatasetModule
from llm_benchmark.utils.enums import DatasetType, Tags, Quality, QuestionType
from llm_benchmark.utils.llm_interface.templates import generation_templates as gen_tmps 
from llm_benchmark.utils.llm_interface.templates import evaluation_templates as eval_tmps 
from llm_benchmark.utils.llm_interface import evaluation_utils as eval_utils 
from llm_benchmark.utils.llm_interface import generation_utils as gen_utils 


class LLMInterfaceModule():
    def __init__(self):
        print(f"Initialized {self.__class__.__name__}")
        pass
    
    def generate_questions(self, 
                           DatasetModule: DatasetModule,
                           params: Optional[Dict[str, Any]] = None,
                           output_path: str = "",
                           batch_size: Optional[int] = 1,
                           ) -> None:
        if params is None:
            print("Warning: No parameters provided for question generation. Using default settings.")
            params = {}
        if output_path == None or output_path == "":
            return

        try:
            ds_endpoint: str = DatasetModule.get_endpoint()
            sub_dir: str = ds_endpoint.split("/")[-2]
        except:
            print("Failed to parse endpoint")
            return
        
        print(f"Generating questions for dataset at endpoint: {sub_dir}")

        output_dirname: str = os.path.dirname(output_path)
        if output_dirname:
            os.makedirs(output_dirname, exist_ok=True)

        dataset: pl.DataFrame = DatasetModule.get_entries()
        question_outputs: List[Any] = [""] * dataset.height

        height: int = dataset.height
        batch_count: int = height // batch_size

        output_buffer: List[str] = []

        for idx in range(batch_count+1):
            batch_slice: List[Dict[str, Any]] = dataset.slice(offset=idx * batch_size, length=batch_size).to_dicts()

            if batch_slice == []:
                continue

            input: List[str] = self._format_batch_input\
                (
                    batch=batch_slice,
                    sub_dir=sub_dir
                )
            raw_outs: List[str] = self.query_model\
                (
                    input,
                    temperature = params.get("temperature", GENERATE_TEMPERATURE),
                    max_tokens = params.get("max_tokens", GENERATE_TOKENS_PER_PROMPT),
                    seed = params.get("seed", 42)
                )
            
            output_buffer.append(raw_outs)

            
        for idx, out in tqdm(enumerate(output_buffer)):
            if out is None:
                out = ""
            entry: Dict[str, Any] = {
            "endpoint_identifier": ds_endpoint, 
            "entry_idx": idx,
            "output": out,
            }
            question_outputs[idx] = entry
        
        if not question_outputs:
            print(f"Failed to generate questions for {DatasetModule.get_endpoint()}")
            return
        
        self.write_outputs(raw=question_outputs, output_path=output_path)

    def _format_single_input(
            self,
            itm: Dict[str, Any],
            params: Dict[str, Any],
            sub_dir: str,
        ) -> List[str]:
        
        if not self.check_if_question_in_filter(itm, params):
            print("Skipping question due to filter settings.")
            return ""
        
        question_type: QuestionType = util.question_type_remap(
            itm.get("question_type", None), 
            row=itm,
            endpoint=sub_dir
            )
        template_method: Optional[Callable] = self._get_template(question_type=question_type)
        if not template_method:
            print(f"Warning: template method for question type {question_type} not set in the configuration file.")
            message_single: str = ""
        else:
            message_single: str = template_method(gen_utils.remap_question_row(itm, sub_dir))
        
        return message_single
        

    def query_model(self, 
                    message: List[str],
                    temperature: float = 0.7,
                    max_tokens: int = 150,
                    batch_size: int = 1,
                    ) -> str:
        raise NotImplementedError("This method should be overridden by subclasses.")

    def _get_template(self, question_type: QuestionType) -> object:
        return question_generation_template_mapping.get(question_type, None)

    def respond_to_questions(self,
                            DatasetModule: DatasetModule,
                            params: Optional[Dict[str, Any]] = None,
                            output_path: str = ""
                            ) -> None:
        if params is None:
            print("Warning: No parameters provided for question generation. Using default settings.")
            params = {}
        
        ds_endpoint: str = DatasetModule.get_endpoint()
        if ds_endpoint is None:
            print("No Endpoint Specified")
            return None    
            
        question_df: pl.DataFrame = DatasetModule.get_questions()
        if question_df is None:
            print(f"No questions linked to dataset {ds_endpoint}.")
            return None
        
        output_dirname: str = os.path.dirname(output_path)
        os.makedirs(output_dirname, exist_ok=True)

        question_outputs: List[Dict[str, Any]] = self._create_empty_outputs(question_df.height)
        
        for idx, row in tqdm(enumerate(question_df.to_dicts())):
            output: str = self._respond_single(row=row, params=params)
            if output is None:
                continue
            print("Query Output:", output)
            question_outputs[idx] = {
                "endpoint_identifier": ds_endpoint, 
                "entry_idx": idx,
                "output": output,
                }
        
        self.write_outputs(raw=question_outputs, output_path=output_path)


    def _create_empty_outputs(self, height: int) -> List[Dict[str, Any]]:
        return [{"endpoint_identifier": None, "entry_idx": None, "output": None} for _ in range(height)]


    def _respond_single(self, row: Dict[str, Any], params: Dict[str, Any]) -> str:
        question: str = row.get("output", "")
        if not question:
            return None
        prompt: Dict[str, Any] = eval_tmps.multichoice(question)
        return self.query_model(
            prompt,
            temperature=params.get("temperature", QUERY_TEMPERATURE),
            max_tokens=params.get("max_tokens", QUERY_TOKENS_PER_PROMPT)
            )

    def write_outputs(self, 
                      raw: List[Dict[str, Any]], 
                      output_path: str
                      ) -> None:
        """Writes the generated questions to a CSV file."""
        if not output_path:
            print(f"Output path doesnt exist.")
            return
        if not raw:
            print(f"Failed to generate questions.")
            return
        
        pl.DataFrame(raw).write_csv(output_path)
        print(f"Questions generated and saved to {output_path}")
    
    # Utility
    def hugging_face_model_load(self,
                                model_name: str,
                                trust_remote_code: Optional[bool] = False,
                                local: Optional[bool] = True,
                                cache_dir: Optional[str] = "/rds/general/user/cp824/ephemeral/huggingface_cache"
                                ) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:        
        print(f"Loading model {model_name}.")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                                  trust_remote_code=trust_remote_code,
                                                  local_files_only=local,
                                                  cache_dir=cache_dir,
                                                  )
        print("Tokenizer loaded successfully.")
        model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                     torch_dtype="auto", 
                                                     device_map="auto", 
                                                     trust_remote_code=trust_remote_code,
                                                     cache_dir=cache_dir,
                                                     local_files_only=local
                                                     )
        try:
            model.generation_config = GenerationConfig.from_pretrained(model_name)
        except Exception as e:
            print(f"Failed to load generation config for {model_name}. Using default config. Error: {e}")
            model.generation_config = GenerationConfig()
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

        return tokenizer, model
    
    def pull_remote_weights(self,
                            model_name: str,
                            trust_remote_code: Optional[bool] = False,
                            cache_dir: Optional[str] = "/rds/general/user/cp824/ephemeral/huggingface_cache",
                            ) -> None:
        self.hugging_face_model_load(model_name=model_name,
                                     trust_remote_code=trust_remote_code, 
                                     local=False,
                                     cache_dir=cache_dir,)
        
    def check_if_question_in_filter(self,
                                    question: Dict[str, Any],
                                    params: Dict[str, Any]
                                    ) -> bool:
        """Check if a question matches the filter parameters."""
        for param_key, (question_key, remap) in params_to_question_mapping.items(): 
            param_item: List[Any] = params.get(param_key, [])
            question_item: Any = remap(question.get(question_key, None), question)

            if param_item and question_item not in param_item:
                return False
        return True