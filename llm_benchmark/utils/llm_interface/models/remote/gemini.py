import os
import json
import time
import logging

import polars as pl

from datetime import datetime
from google import genai
from google.genai import types
from typing import Any, Dict, List, Optional, Tuple

from llm_benchmark.utils.llm_interface.query_core import LLMInterfaceModule
from llm_benchmark.utils.llm_interface import generation_utils as gen_utils
from llm_benchmark.config import CACHE_PATH, ENABLE_API_CALLS, BATCH_POLLING_FREQUENCY_SECONDS, DATABASE_PATH, OPENAI_REASONING_EFFORT, MANUAL_REASONING


class GeminiInterfaceModule(LLMInterfaceModule):
    def __init__(self,
                 model_name: str = "",
                 api_key: str = ""
                 ) -> None:
        super().__init__()

        if api_key is None or api_key == "":
            raise ValueError("Gemini API key must be provided.")
        if model_name is None or model_name == "":
            raise ValueError("Error: Gemini model must be specified.")
        
        print(f"Initialized Gemini model: {model_name}.")
        
        self.model_name = model_name
        self.client = self.initialize_client(api_key)


    def initialize_client(
            self, 
            key: str
            ) -> genai.Client:
        return genai.Client(api_key=key)
    
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
            case -1:
                response: List[Tuple[str, Optional[str], List[Dict[str, Any]], Dict[str, Any]]] = self._batch(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    endpoint=endpoint,

                    logger=logger,
                )
                return response
            case _:
                responses = []
                for itm in messages:
                    response: Tuple[str, Optional[str], List[Dict[str, Any]], Dict[str, Any]] = self._single(
                        message=itm,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        seed=seed,

                        logger=logger
                    )
                    logger.info(response)
                    responses.append(response)
                return responses

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
            path=input_path,
            temperate=temperature,
        )

        if file is None:
            return self._flush_batch(path=input_path)
        if not ENABLE_API_CALLS:
            logger.warning(f"API calls are disabled for: {self.model_name}, enable ENABLE_API_CALLS in config.")
            return []
        
        batch_job = self.client.batches.create(
            model=self.model_name,
            src=file.name,
            config={'display_name': f"research_batch_{int(time.time())}"}
        )

        logger.info(f"Submitted batch job: {batch_job.name}")
    
        final_batch = self._poll_batch(
            client=self.client,
            batch_job=batch_job,
            logger=logger,
        )

        if final_batch is None:
            logger.warning(f"Failed to generate batch for endpoint: {endpoint}.")
            return []
        
        output_basename: str = os.path.join(CACHE_PATH, "jsonl", "outputs", model)
        os.makedirs(output_basename, exist_ok=True)
        file_name: str = f"{dt}_{model}_{endpoint.replace("/", "_")}_{len(os.listdir(output_basename))}.jsonl"

        output_path: str = os.path.join(output_basename, file_name)
        
        file_handle = self.client.files.get(name=final_batch.output_uri)

        with open(output_path, "wb") as f:
            f.write(file_handle.content)
            
        return self._reconstruct_data(
            file_path=output_path,
            original_messages=messages,
            logger=logger,
            temperature=temperature
        )


    def _poll_batch(
        self,
        client: genai.Client,
        batch_job,
        logger: logging.Logger
    ):
        seconds: int = 0
        while True:
            job = client.batches.get(name=batch_job.name)
            if job.state in ["SUCCEEDED", "FAILED"]:
                if job.state == "SUCCEEDED":
                    return job
                logger.info(f"Batch failed with status: {job.state}, message: {getattr(job, 'error', 'No specific error message provided.')}.")
                return None
                
            time.sleep(BATCH_POLLING_FREQUENCY_SECONDS)
            seconds += BATCH_POLLING_FREQUENCY_SECONDS
            logger.info(f"Current time: {str(seconds)}")
            
    def _reconstruct_data(
            self,
            file_path: str,
            original_messages: List[List[Dict[str, Any]]],
            logger: logging.Logger,
            temperature: float
    ) -> List[Tuple[str, Optional[str], Dict[str, Any]]]:
        
        outs = [()] * len(original_messages)
        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                response = data.get("response", {})
                candidates = response.get("candidates", [])
                
                content = ""
                reasoning = None
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    content = "".join([p.get("text", "") for p in parts])
                
                usage = response.get("usageMetadata", {})
                others = {
                    "usage.total_tokens": usage.get("totalTokenCount", 0),
                    "temperature": temperature,
                    "jsonl.output": file_path,
                }
                outs[i] = (content, reasoning, original_messages[i], others)
        return outs

        
    def _flush_batch(self, path: str) -> List[Tuple[str, Optional[str], Dict[str, Any]]]:
        if os.path.exists(path):
            os.remove(path)
        return []
    
    def _create_jsonl(
            self, 
            messages: List[List[Dict[str, Any]]],
            max_tokens: int,
            temperate: float,
            path: str,
            ) -> Optional[object]:
        with open(path, mode="w+") as file:
            for idx, itm in enumerate(messages):
                if not isinstance(itm, list):
                    continue
                schema: Dict[str, Any] = self._batch_schema\
                    (
                        idx=idx,
                        message=itm,
                        max_tokens=max_tokens,
                        temperature=temperate,
                    )
                file.write(f"{json.dumps(schema)}\n")
        
        return self.client.files.upload(file=path, config={"mime_type": "application/x-jsonlines"})

    def _batch_schema(
            self,
            idx: int,
            message: List[Dict[str, Any]],

            max_tokens: int,
            temperature: float,
            
            ) -> Dict[str, Any]:

        return\
            {
                "request": {
                    "contents": self._reformat_message(message),
                    "generationConfig": {
                        "maxOutputTokens": max_tokens,
                        "temperature": temperature # Add other config params here
                    }
                }
            }
    
    def _reformat_message(
            self,
            message: List[Dict[str, Any]]
            ) -> List[Dict[str, Any]]:
        outs: List[Dict[str, Any]] = []
        for itm in message:
            role = "model" if itm["role"] == "assistant" else "user"
            part = types.Part.from_text(text=itm.get("content", ""))
            content_obj = types.Content(
                role=role,
                parts=[part]
            )
            outs.append(content_obj)
        return outs

    def _single(
        self,
        message: List[Dict[str, Any]],
        
        max_tokens: int,
        temperature: float,
        seed: int,

        logger: logging.Logger,

        max_tries: int = 3,
        current_try: int = 0,
    ) -> Tuple[str, Optional[str], List[Dict[str, Any]], Dict[str, Any]]:
        
        def error_result(msg: str) -> Tuple[str, Optional[str], List[Dict[str, Any]], Dict[str, Any]]:
            return ("", None, message, {"error": msg})
        
        if message is None or message == [] or message == "":
            logger.warning(f"Message is empty, skipping.")
            return error_result(msg="Message is empty")
        if not ENABLE_API_CALLS:
            logger.warning(f"API calls are disabled for: {self.model_name}, enable ENABLE_API_CALLS in config.")
            return error_result(msg="API calls are disabled via config, enable ENABLE_API_CALLS.")
        
        try:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=self._reformat_message(message),
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                        seed=seed
                    )
                )
            except google.genai.errors.ServerError: 
                if current_try > max_tries:
                    return error_result(msg="Exceeded the maximum number of tries!")
                return self._single(
                    message=message,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                    logger=logger,
                    max_tries=max_tries,
                    current_try=current_try + 1
                )

        except Exception as e:
            return error_result(msg=str(e))
        
        content = response.text
        usage = response.usage_metadata
        others = {
            "usage.total_tokens": usage.total_token_count,
            "temperature": temperature,
        }
        return (content or "", None, message, others)