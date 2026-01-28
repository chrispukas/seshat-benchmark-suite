import os
import polars as pl

from openai import OpenAI
from typing import Any, Dict, List, Optional, Tuple

from llm_benchmark import config as cfg

from llm_benchmark.utils import utility as util
from llm_benchmark.utils.dataset import DatasetModule
from llm_benchmark.utils.enums import DatasetType, Tags, Quality, QuestionType
from llm_benchmark.utils.llm_interface.query_core import QuestionGenerationModule

from llm_benchmark.utils.llm_interface import generation_utils as gen_utils 



class ChatGPTQuestionGenerationModule(QuestionGenerationModule):
    def __init__(self,
                 model_name: str = "gpt-3.5-turbo",
                 key: str = ""
                 ) -> None:
        
        if key is None or key == "":
            raise ValueError("OpenAI API key must be provided.")
        
        super().__init__()

        self.model_name = model_name
        self.client = self.initialize_client(key)


    def initialize_client(self, 
                          key: str
                          ) -> OpenAI:
        return OpenAI(api_key=key)
    
    def query_model(self, 
                    message: Dict[str, Any],
                    temperature: float = 0.7,
                    max_tokens: int = 150
                    ) -> str:
        print(f"Querying model {self.model_name} with message: {message}")
        return ""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response    

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