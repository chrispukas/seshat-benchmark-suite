from email.mime import message
import os
import polars as pl

from typing import Any, Dict, List, Optional, Tuple

from llm_benchmark import config as cfg

from llm_benchmark.utils.llm_interface.query_core import QuestionGenerationModule
from llm_benchmark.utils.llm_interface import generation_utils as gen_utils 
from llm_benchmark.utils import utility as util

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


class QwenQuestionGenerationModule(QuestionGenerationModule):
    def __init__(self,
                 model_name: str = "Qwen/Qwen-7B-Chat",
                 trust_remote_code: Optional[bool] = True,
                 local: Optional[bool] = True,
                 ) -> None:
        
        if model_name is None:
            raise ValueError("Model name must be provided.")
        
        super().__init__()

        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.local = local
        self.tokenizer, self.model = self.initialize_client()


    def initialize_client(self, 
                          ) -> Any:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            trust_remote_code=True,
            bf16=True
        ).eval()


        return tokenizer, model
    
    def query_model(self, 
                    message: Dict[str, Any],
                    temperature: float = 0.7,
                    max_tokens: int = 300
                    ) -> str:
        
        print(f" /// STARTOF ///")
        print(f"Querying Qwen model, message: {message}")
        print(f" /// ENDOF ///")


        prompt = self.collapse_prompt(message)
        response, _ = self.model.chat(self.tokenizer, prompt, history=None)

        print(f" /// STARTOF ///")
        print(f"Output: {response}")
        print(f" /// ENDOF ///")

        return response
    
    def collapse_prompt(self, messages: List[Dict[str, str]]) -> str:
        prompt = ""
        for msg in messages:
            role = msg.get("role", "").upper()
            content = msg.get("content", "")
            prompt += f"{role}:\n{content}\n\n"
        prompt += "ASSISTANT:\n"
        return prompt