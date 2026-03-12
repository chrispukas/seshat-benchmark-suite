import os
import polars as pl

from typing import Any, Dict, List, Optional, Tuple

from llm_benchmark import config as cfg

from llm_benchmark.utils.llm_interface.query_core import LLMInterfaceModule
from llm_benchmark.utils.llm_interface import generation_utils as gen_utils 
from llm_benchmark.utils import utility as util


class DeepSeekInterfaceModule(LLMInterfaceModule):
    def __init__(self,
                 model_name: str = "deepseek-moe-16b-base",
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
        return self.hugging_face_model_load(self.model_name, local=self.local, trust_remote_code=self.trust_remote_code)
    
    def query_model(self, 
                    message: Dict[str, Any],
                    temperature: float = 0.7,
                    max_tokens: int = 300
                    ) -> str:
        
        print(f" /// STARTOF ///")
        print(f"Querying DeepSeek model, message: {message}")
        print(f" /// ENDOF ///")

        prompt = util.collapse_prompt(message)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.model.generate(**inputs.to(self.model.device), 
                                      max_new_tokens=max_tokens, 
                                      temperature=temperature)


        outputs: str = self.tokenizer.decode(outputs[0], skip_special_tokens=True); 

        print(f" /// STARTOF ///")
        print(f"Output: {outputs}")
        print(f" /// ENDOF ///")

        return outputs