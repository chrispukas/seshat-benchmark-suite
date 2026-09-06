import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_benchmark.utils import utility as util
from llm_benchmark.utils.llm_interface.query_core import LLMInterfaceModule


class OlmoInterfaceModule(LLMInterfaceModule):
    def __init__(
        self,
        model_name: str = "allenai/Olmo-3-1025-7B",
        trust_remote_code: Optional[bool] = True,
        local: Optional[bool] = True,
    ) -> None:
        if not model_name:
            raise ValueError("Model name must be provided.")

        super().__init__(model_name=model_name)

        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.local = local
        self.tokenizer, self.model = self.initialize_client()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

    def initialize_client(self) -> Tuple[Any, Any]:
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local,
            torch_dtype="auto",
            device_map="auto",
        )

        return tokenizer, model

    def query_model(
        self,
        messages: List[List[Dict[str, Any]]],
        logger: logging.Logger,
        temperature: float = 0.7,
        max_tokens: int = 300,
        seed: int = 42,
        endpoint: str = "",
    ) -> List[
        Tuple[str, Optional[str], List[Dict[str, Any]], Dict[str, Any]]
    ]:
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        outputs: List[
            Tuple[str, Optional[str], List[Dict[str, Any]], Dict[str, Any]]
        ] = []

        for message in messages:
            if not message:
                outputs.append((
                    "",
                    None,
                    [],
                    {"error": "Message is empty"},
                ))
                continue

            try:
                prompt = util.collapse_prompt(message)

                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    return_token_type_ids=False,
                )

                device = next(self.model.parameters()).device
                inputs = {
                    key: value.to(device)
                    for key, value in inputs.items()
                }

                generation_kwargs: Dict[str, Any] = {
                    "max_new_tokens": max_tokens,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "do_sample": temperature > 0,
                }

                if temperature > 0:
                    generation_kwargs.update({
                        "temperature": temperature,
                        "top_p": 0.7,
                    })

                with torch.inference_mode():
                    generated = self.model.generate(
                        **inputs,
                        **generation_kwargs,
                    )

                prompt_length = inputs["input_ids"].shape[-1]
                generated_tokens = generated[0, prompt_length:]

                answer = self.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True,
                ).strip()

                metadata = {
                    "model": self.model_name,
                    "endpoint": endpoint,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

                outputs.append((answer, None, message, metadata))

            except Exception as exc:
                logger.exception("OLMo inference failed: %s", exc)
                outputs.append((
                    "",
                    None,
                    message,
                    {
                        "model": self.model_name,
                        "endpoint": endpoint,
                        "error": str(exc),
                    },
                ))

        return outputs