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