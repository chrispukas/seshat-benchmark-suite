import os
import json
import time
import random
import logging

import polars as pl

from click import File
from datetime import datetime
from openai import OpenAI, RateLimitError, BadRequestError
from typing import Any, Dict, List, Optional, Tuple

from llm_benchmark.utils.llm_interface.query_core import LLMInterfaceModule
from llm_benchmark.utils.llm_interface import generation_utils as gen_utils
from llm_benchmark.config import CACHE_PATH, ENABLE_API_CALLS, BATCH_POLLING_FREQUENCY_SECONDS, DATABASE_PATH, OPENAI_REASONING_EFFORT, MANUAL_REASONING


class ChatGPTInterfaceModule(LLMInterfaceModule):
    def __init__(self,
                 model_name: str = "",
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
                    temperature=temperature,
                    endpoint=endpoint,

                    logger=logger,
                )
                return response

    def _batch(
            self, 
            messages: List[List[Dict[str, Any]]],
            max_tokens: int,
            temperature: float,
            endpoint: str,
            logger: logging.Logger,
            ) -> List[Tuple[str, Optional[str], List[Dict[str, Any]], Dict[str, Any]]]:
        dt: str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        model: str = self.model_name.replace("/", "_")
        input_path: str = os.path.join(DATABASE_PATH, "jsonl", "inputs", model, f"{dt}_{model}_{endpoint.replace("/", "_")}.jsonl")
        os.makedirs(os.path.dirname(input_path), exist_ok=True)
        file = self._create_jsonl(
            messages=messages,
            max_tokens=max_tokens,
            path=input_path
        )

        if file is None:
            return self._flush_batch(path=input_path)
        if not ENABLE_API_CALLS:
            logger.warning(f"API calls are disabled for: {self.model_name}, enable ENABLE_API_CALLS in config.")
            return []
        
        batch = self.client.batches.create(
            input_file_id=file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
    
        final_batch = self._poll_batch(
            client=self.client,
            batch_id=batch.id,
            logger=logger,
        )

        if final_batch is None:
            logger.warning(f"Failed to generate batch for endpoint: {endpoint}.")
            return []
        
        output_basename: str = os.path.join(CACHE_PATH, "jsonl", "outputs", model)
        os.makedirs(output_basename, exist_ok=True)
        file_name: str = f"{dt}_{model}_{endpoint.replace("/", "_")}_{len(os.listdir(output_basename))}.jsonl"

        output_path: str = os.path.join(output_basename, file_name)
        
        file_response = self.client.files.content(final_batch.output_file_id)
        with open(output_path, "wb") as f:
            f.write(file_response.content)
        return self._reconstruct_data\
            (
                file_path=output_path,
                original_messages=messages,
                logger=logger,
                temperature=temperature
            )


    def _poll_batch(
        self,
        client: OpenAI,
        batch_id: str,
        logger: logging.Logger
    ):
        seconds: int = 0
        while True:
            batch = client.batches.retrieve(batch_id=batch_id)
            if batch.status in ["completed", "failed", "expired", "cancelled"]:
                if batch.status == "completed":
                    return batch
                logger.info(f"Batch failed with status: {batch.status}, message: {batch.errors}.")
                return None
                
            time.sleep(BATCH_POLLING_FREQUENCY_SECONDS)
            seconds += BATCH_POLLING_FREQUENCY_SECONDS
            logger.info(f"Current time: {str(seconds)}, polling batch id: {batch_id}")
            
    def _reconstruct_data(
            self,
            file_path: str,
            original_messages: List[List[Dict[str, Any]]],
            logger: logging.Logger,
            temperature: float
    ) -> List[Tuple[str, Optional[str], Dict[str, Any]]]:
        
        with open(file_path, "r") as f:
            lines: List[str] = f.readlines()
            outs: List[Tuple[str, Optional[str], Dict[str, Any]]] = [()] * len(original_messages)

            for line in lines:
                json_line: Dict[str, Any] = json.loads(line)
                idx: int = int(json_line["custom_id"])

                if idx >= len(outs):
                    logger.error(f"Index {idx} out of range! Check if batch file matches input.")
                    continue

                response: Dict[str, Any] = json_line["response"]

                if not response or response.get("status_code") != 200:
                    outs[idx] = (
                        ("", None, [{}], {"error": "API failed"})
                    )
                    continue

                body = response["body"]
                choices = body.get("choices", [])
                message_data = choices[0]["message"] if choices else {}

                content = message_data.get("content", "")
                reasoning = message_data.get("reasoning_content")

                usage = body.get("usage", {})
                others: Dict[str, Any] = {
                    "usage.total_tokens": usage.get("total_tokens", 0),
                    "usage.prompt_tokens": usage.get("prompt_tokens", 0),
                    "usage.cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
                    "temperature":  temperature,
                    "reasoning_effort": OPENAI_REASONING_EFFORT,
                    "manual_reasoning": MANUAL_REASONING,
                    "jsonl.output": file_path,
                }
                
                outs[idx] = (content, reasoning, original_messages[idx], others)
        return outs

    
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
                if not isinstance(itm, list):
                    continue
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

        token_field_name: str = "max_tokens"

        if self.model_name.startswith("gpt-5") or self.model_name.startswith("o"):
            token_field_name = "max_completion_tokens"

        body: Dict[str, Any] = {
            "model": str(self.model_name),
            "messages": message,
            token_field_name: max_tokens,
        }

        if not MANUAL_REASONING:
            body["reasoning_effort"] = OPENAI_REASONING_EFFORT
            
        return \
            {
                "custom_id": str(idx),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body
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
            return ("", None, [{}], {"error": "Message is empty"})
        if not ENABLE_API_CALLS:
            logger.warning(f"API calls are disabled for: {self.model_name}, enable ENABLE_API_CALLS in config.")
            return ("", None, [{}], {"error": "API calls are disabled via config, enable ENABLE_API_CALLS."})
        
        
        response = self._create_with_retry(
            max=3,
            curr=0,
            message=message,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed
        )

        if response is None:
            return ("", None, [{}], {"error": "API Error: Exceeded rate limit."})

        choice = response.choices[0].message

        content: Optional[str] = choice.content
        reasoning = getattr(choice, 'reasoning_content', None)

        cached: int = 0
        try:
            cached: int = getattr(response.usage, 'prompt_tokens_details', {}).get('cached_tokens', 0)
        except:
            cached: int = -1

        others: Dict[str, Any] = \
            {
                "usage.total_tokens": response.usage.total_tokens,
                "usage.prompt_tokens": response.usage.prompt_tokens,
                "usage.cached_tokens": cached,
                "temperature": temperature,
                "reasoning_effort": OPENAI_REASONING_EFFORT,
                "manual_reasoning": MANUAL_REASONING,
                "jsonl.output": "",
            }
        
        logger.info(
            f"    Metrics: "
            f"total_token usage: {others['usage.total_tokens']}, "
            f"prompt_token usage: {others['usage.prompt_tokens']}"
            f"cached_token usage: {others['usage.cached_tokens']}"
        )

        return (content or "", reasoning, message, others)
    

    def _create_with_retry(
            self,
            max: int, 
            curr: int,
            message: List[Dict[str, Any]],

            max_tokens: int,
            temperature: float,
            seed: int
            ):
        try:
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=message,

                max_completion_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            )
        except (RateLimitError, BadRequestError):
            if curr > max:
                return None
            else:
                delay = min(60, (2 ** curr) + random.uniform(0, 1))
                time.sleep(delay)

                return self._create_with_retry(
                    max=max,
                    curr=curr+1,
                    
                    message=message,
                    seed=seed,

                    max_tokens=max_tokens,
                    temperature=temperature
                )
