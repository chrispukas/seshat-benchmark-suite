import os
import json
import time
import logging

import polars as pl

from click import File
from openai import OpenAI
from typing import Any, Dict, List, Optional, Tuple

from llm_benchmark.utils.llm_interface.query_core import LLMInterfaceModule
from llm_benchmark.config import CACHE_PATH, ENABLE_API_CALLS

class ChatGPTInterfaceModule(LLMInterfaceModule):
    def __init__(self,
                 model_name: str = "gpt-3.5-turbo",
                 api_key: str = ""
                 ) -> None:
        super().__init__()

        if api_key is None or api_key == "":
            raise ValueError("OpenAI API key must be provided.")
        if model_name is None or model_name == "":
            raise ValueError("Error: OpenAI model must be specified.")
        
        print(f"Initialized OpenAI model: {model_name}.")
        
        self.model_name = model_name
        self.client = self.initialize_client(api_key)


    def initialize_client(
            self, 
            key: str
            ) -> OpenAI:
        return OpenAI(api_key=key)
    
    def query_model(
            self, 
            messages: List[List[Dict[str, Any]]],

            logger: logging.Logger,
            
            max_tokens: int = 300,
            temperature: float = 0.7,

            seed: int = 42,
            endpoint: str = "",
            ) -> List[Tuple[str, Optional[str], List[Dict[str, Any]], Dict[str, Any]]]:
        logger.info(f"    Querying model {self.model_name} with {len(messages)} message(s), temperature: {temperature}")

        match len(messages):
            case 0:
                return []
            case 1:
                response: Tuple[str, Optional[str], List[Dict[str, Any]], Dict[str, Any]] = self._single(
                    message=messages[0],

                    max_tokens=max_tokens,
                    temperature=temperature,

                    seed=seed,

                    logger=logger
                )
                logger.info(response)
                return [response]
            case _:
                response: List[Tuple[str, Optional[str], List[Dict[str, Any]], Dict[str, Any]]] = self._batch(
                    messages=messages,
                    max_tokens=max_tokens,
                    endpoint=endpoint,
                )
                return response

    def _batch(
            self, 
            messages: List[List[Dict[str, Any]]],
            max_tokens: int,
            endpoint: str,
            ) -> List[Tuple[str, Optional[str], Dict[str, Any]]]:
        path: str = os.path.join(CACHE_PATH, f"tmp_chatgpt_{endpoint}.jsonl")
        file = self._create_jsonl(
            messages=messages,
            max_tokens=max_tokens,
            path=path
        )

        if file is None:
            return self._flush_batch(path=path)

        batch = self.client.batches.create(
            input_file_id=file.id,
            endpoint="/v1/chat/completions",
            completion_window="1h"
        )
    
        self.client.batches.retrieve(batch.id)

        os.remove(path)
        raise NotImplementedError("Batch processing is not fully implemented yet. Please use single message queries for now.")
    
    def _flush_batch(self, path: str) -> List[Tuple[str, Optional[str], Dict[str, Any]]]:
        if os.path.exists(path):
            os.remove(path)
        return []
    
    def _create_jsonl(
            self, 
            messages: List[List[Dict[str, Any]]],
            max_tokens: int,
            path: str,
            ) -> Optional[File]:
        with open(path, mode="w+") as file:
            for idx, itm in enumerate(messages):
                schema: Dict[str, Any] = self._batch_schema\
                    (
                        idx=idx,
                        message=itm,
                        max_tokens=max_tokens
                    )
                file.write(f"{json.dumps(schema)}\n")
        
        with open(path, "rb") as f:
            file = self.client.files.create(
                file=f,
                purpose="batch"
            )

        return file



    def _batch_schema(
            self,
            idx: int,
            message: List[Dict[str, Any]],

            max_tokens: int,
            
            ) -> Dict[str, Any]:
        return \
            {
                "custom_id": str(id),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": \
                {
                    "model": str(self.model_name),
                    "messages": message,
                    "max_completion_tokens": max_tokens
                }
            }
    
    def _single(
        self,
        message: List[Dict[str, Any]],
        
        max_tokens: int,
        temperature: float,
        seed: int,

        logger: logging.Logger,
    ) -> Tuple[str, Optional[str], List[Dict[str, Any]], Dict[str, Any]]:
        
        if message is None or message == []:
            logger.warning(f"Message is empty, skipping.")
            return ("", None, [{}], {})
        if not ENABLE_API_CALLS:
            logger.warning(f"API calls are disabled for: {self.model_name}, enable ENABLE_API_CALLS in config.")
            return ("", None, [{}], {})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message,

            max_completion_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        )
        reasoning: Optional[str] = getattr(message, 'reasoning_content', None)
        content: Optional[str] = response.choices[0].message.content
        others: Dict[str, Any] = \
            {
                "usage.total_tokens": response.usage.total_tokens,
                "usage.prompt_tokens": response.usage.prompt_tokens,
            }
        
        logger.info(
            f"    Metrics:"
            f"total_token usage: {others['usage.total_tokens']}, "
            f"prompt_token usage: {others['usage.prompt_tokens']}"
        )

        return (content or "", reasoning, message, others)