from email.mime import message
import os
import polars as pl

from typing import Any, Dict, List, Optional, Tuple, List

from llm_benchmark import config as cfg

from llm_benchmark.utils.llm_interface.query_core import LLMInterfaceModule
from llm_benchmark.utils.llm_interface import generation_utils as gen_utils 
from llm_benchmark.utils import utility as util

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


class QwenInterfaceModule(LLMInterfaceModule):
    def __init__(self,
                 model_name: str = "Qwen/Qwen-7B-Chat",
                 trust_remote_code: Optional[bool] = True,
                 local: Optional[bool] = True,
                 pull_model: Optional[bool] = False,
                 test_mode: Optional[bool] = False,
                 ) -> None:
        
        if model_name is None:
            raise ValueError("Model name must be provided.")
        
        super().__init__()

        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.local = local

        if test_mode:
            return

        try:
            self.tokenizer, self.model = self._initialize_client(bf16=True, pull_model=pull_model)
        except:
             self.tokenizer, self.model = self._initialize_client(pull_model=pull_model)

    def _initialize_client(self,
                          bf16: bool = False,
                          pull_model: bool = False
                          ) -> Any:
        device_map: str = "meta" if pull_model else "auto"
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, local_files_only=self.local)
        if bf16:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=device_map,
                trust_remote_code=True,
                bf16=True,
                local_files_only=self.local
            ).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=device_map,
                trust_remote_code=True,
                local_files_only=self.local
            ).eval()

        return tokenizer, model
    
    def query_model(self, 
                    messages: List[str],
                    temperature: float = 0.7,
                    max_tokens: int = 300,
                    seed: int = 42,
                    ) -> str:
        print(f"\n\n\n")
        print(f"Querying Qwen model, message: {messages}")
        try:
            response = self.new_chat(
                temperature=temperature, 
                max_tokens=max_tokens, 
                batch_messages=messages, 
                )
        except:
            response = self.old_chat(
                messages=messages)
        print(f"       Output: {response}")
        return response
    
    def old_chat(
            self, 
            messages: List[str]
            ):
        prompt = util.collapse_prompt(messages[0])
        response, _ = self.model.chat(self.tokenizer, prompt, history=None)
        return response

    def new_chat(
            self, 
            temperature: Optional[int], 
            max_tokens: Optional[int], 
            batch_messages: List[str], 
            ):
        templated_texts: List[Any] = [
            self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            for messages in batch_messages
        ]
        inputs = self.tokenizer(
            templated_texts,
            return_tensors="pt",
            padding=True,
            return_dict=True
        ).to(self.model.device)
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_tokens,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True if temperature > 0 else False
            )
        input_length = inputs["input_ids"].shape[-1]
        
        return [
            self.tokenizer.decode(output[input_length:], skip_special_tokens=True).strip()
            for output in outputs
        ]

