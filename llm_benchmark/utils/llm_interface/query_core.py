import torch
import polars as pl

from typing import Any, Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from llm_benchmark.config import params_to_question_mapping

from llm_benchmark.utils import utility as util
from llm_benchmark.utils.dataset import DatasetModule
from llm_benchmark.utils.enums import DatasetType, Tags, Quality, QuestionType
from llm_benchmark.utils.llm_interface import generation_utils as gen_utils 



class QuestionGenerationModule():
    def __init__(self):
        print(f"Initialized {self.__class__.__name__}")
        pass
    
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
    
    def generate_questions(self, 
                           DatasetModule: DatasetModule,
                           params: Optional[Dict[str, Any]] = None
                           ) -> None:
        if params is None:
            print("Warning: No parameters provided for question generation. Using default settings.")
            params = {}

        if DatasetModule.get_endpoint() is None:
            print("No Endpoint Specified")
            return None

        sub_dir = DatasetModule.get_endpoint().split("/")[-2]
        dataset: pl.DataFrame = DatasetModule.get_entries()

        for idx, row in enumerate(dataset.iter_rows(named=True)):
            if not self.check_if_question_in_filter(row, params):
                print("Skipping question due to filter settings.")
                continue

            question_type: QuestionType = util.question_type_remap(
                row.get("question_type", None), 
                row=row,
                endpoint=sub_dir
                )
            print(question_type)
            message: Dict[str, Any] = {}

            match question_type:
                case QuestionType.MULTIPLE_CHOICE:
                    message: Dict[str, Any] = gen_utils.multichoice_question(row)
                case QuestionType.RANGE:
                    message: Dict[str, Any] = gen_utils.range_question(row)
                case _:
                    print(f"Unimplemented question type for row {idx}, with type {question_type}, skipping.")
                    continue

            self.query_model(message,
                             temperature=params.get("temperature", 0.7),
                             max_tokens=params.get("max_tokens", 150)
                             )
            
    def generate_questions(self, 
                           DatasetModule: Any) -> None:
        raise NotImplementedError("This method should be overridden by subclasses.")
    def query_model(self, 
                    message: Dict[str, Any],
                    temperature: float = 0.7,
                    max_tokens: int = 150
                    ) -> str:
        raise NotImplementedError("This method should be overridden by subclasses.")

    
    def hugging_face_model_load(self,
                                model_name: str,
                                trust_remote_code: Optional[bool] = False
                                ) -> Any:
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                     torch_dtype=torch.bfloat16, 
                                                     device_map="auto", 
                                                     trust_remote_code=trust_remote_code)
        model.generation_config = GenerationConfig.from_pretrained(model_name)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

        return tokenizer, model