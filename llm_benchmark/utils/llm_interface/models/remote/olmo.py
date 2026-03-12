import os
import polars as pl

from typing import Any, Dict, List, Optional, Tuple

from llm_benchmark import config as cfg

from llm_benchmark.utils.llm_interface.query_core import LLMInterfaceModule
from llm_benchmark.utils.llm_interface import generation_utils as gen_utils 
from llm_benchmark.utils import utility as util

from transformers import AutoTokenizer, AutoModelForCausalLM


class OlmoInterfaceModule(LLMInterfaceModule):
    def __init__(self,
                 model_name: str = "allenai/Olmo-3-1025-7B",
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
        
        model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                     trust_remote_code=self.trust_remote_code,
                                                     local_files_only=self.local)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name,
                                                  trust_remote_code=self.trust_remote_code,
                                                  local_files_only=self.local,)
        return tokenizer, model
    
    def query_model(self, 
                    message: Dict[str, Any],
                    temperature: float = 0.7,
                    max_tokens: int = 300
                    ) -> str:
        
        print(f" /// STARTOF ///")
        print(f"Querying Olmo model, message: {message}")
        print(f" /// ENDOF ///")

        prompt = util.collapse_prompt(message)
        inputs = self.tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)

        response = self.model.generate(**inputs.to(self.model.device), 
                                      max_new_tokens=max_tokens, 
                                      temperature=temperature,
                                      do_sample=True,
                                      top_k=0,
                                      top_p=0.7,
                                      )
        outputs: str = self.tokenizer.batch_decode(response, skip_special_tokens=True)[0];

        print(f" /// STARTOF ///")
        print(f"Output: {outputs}")
        print(f" /// ENDOF ///")

        return outputs