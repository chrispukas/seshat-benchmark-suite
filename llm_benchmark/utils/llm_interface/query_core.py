import os
import torch
import polars as pl

from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from llm_benchmark.config import params_to_question_mapping

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
                           output_path: str = ""
                           ) -> None:
        if params is None:
            print("Warning: No parameters provided for question generation. Using default settings.")
            params = {}

        if DatasetModule.get_endpoint() is None:
            print("No Endpoint Specified")
            return None

        output_dirname: str = os.path.dirname(output_path)
        os.makedirs(output_dirname, exist_ok=True)


        sub_dir = DatasetModule.get_endpoint().split("/")[-2]
        print(f"Generating questions for dataset at endpoint: {sub_dir}")
        dataset: pl.DataFrame = DatasetModule.get_entries()

        output_df: pl.DataFrame = pl.DataFrame({})
        question_outputs: List[Any] = [""] * dataset.height

        for idx, row in tqdm(enumerate(dataset.iter_rows(named=True))):
            if not self.check_if_question_in_filter(row, params):
                print("Skipping question due to filter settings.")
                continue
            
            question_type: QuestionType = util.question_type_remap(
                row.get("question_type", None), 
                row=row,
                endpoint=sub_dir
                )
            message: Dict[str, Any] = {}

            remap_row: Dict[str, Any] = gen_utils.remap_question_row(row, sub_dir)
            match question_type:
                case QuestionType.MULTIPLE_CHOICE:
                    message: Dict[str, Any] = gen_tmps.multichoice_question(remap_row)
                case QuestionType.RANGE:
                    print(f"Skipping RANGE question for row {idx}.")
                    continue
                    message: Dict[str, Any] = gen_tmps.range_question(remap_row)
                case _:
                    print(f"Unimplemented question type for row {idx}, with type {question_type}, skipping.")
                    continue

            output: str = self.query_model(message,
                             temperature=params.get("temperature", 0.7),
                             max_tokens=params.get("max_tokens", 150)
                             )
            print("Query Output:", output)
            question_outputs[idx] = {
                "endpoint_identifier": DatasetModule.get_endpoint(), 
                "entry_idx": idx,
                "output": output,
                }
        
        if not question_outputs:
            print(f"Failed to generate questions for {DatasetModule.get_endpoint()}")
            return
        
        self.write_outputs(question_outputs, output_df)

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

        question_outputs: List[Any] = [""] * question_df.height 
        
        for idx, row in enumerate(question_df.iter_rows(named=True)):
            question: str = row.get("output", "")
            if not question:
                print(f"No question found in row {idx}, skipping.")
                continue
            prompt: Dict[str, Any] = eval_tmps.multichoice(question)
            flatttened_prompt: str = self.flatten_prompt(prompt)
            output: str = self.query_model(flatttened_prompt,
                             temperature=params.get("temperature", 0.7),
                             max_tokens=params.get("max_tokens", 150)
                             )
            print("Query Output:", output)
            question_outputs[idx] = {
                "endpoint_identifier": ds_endpoint, 
                "entry_idx": idx,
                "output": output,
                }
        
        self.write_outputs(question_outputs, output_path=output_path)

    def write_outputs(self, 
                      raw: Dict[str, Any], 
                      output_path: str
                      ) -> None:
        """Writes the generated questions to a CSV file."""
        if not output_path:
            print(f"Output path doesnt exist.")
            return
        if not raw:
            print(f"Failed to generate questions.")
            return
        
        output_df = output_df.vstack(pl.DataFrame(raw))
        output_df.write_csv(output_path)
        print(f"Questions generated and saved to {output_path}")
        

    def query_model(self, 
                    message: Dict[str, Any],
                    temperature: float = 0.7,
                    max_tokens: int = 150
                    ) -> str:
        raise NotImplementedError("This method should be overridden by subclasses.")
        
    
    def flatten_prompt(self, messages: List[Dict[str, str]]) -> str:
        prompt: List[str] = []
        for msg in messages:
            role = msg.get("role", "").upper()
            content = msg.get("content", "").strip()
            prompt.append(f"{role}:\n{content}")
        return "\n\n".join(prompt) + "\n\nASSISTANT:\n"

    
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
        model.generation_config = GenerationConfig.from_pretrained(model_name)
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