import io
import re
import random
import logging
import torch
import numpy as np

from typing import Dict, Optional, Any, List
from llm_benchmark.utils.enums import Tags, Quality, QuestionType


def quality_remap(quality: str, 
                  row: Dict,
                  logger: Optional[logging.Logger],
                  ) -> Quality:
    """Remap tag strings to Quality enum values."""

    dict_map: Dict[str, Quality] = {
        "PRESENT": Quality.PRESENT,
        "ABSENT": Quality.ABSENT,
        "A~P": Quality.TRANSITIONAL_A_TO_P,
        "P~A": Quality.TRANSITIONAL_P_TO_A
    }

    return dict_map.get(quality, Quality.UNKNOWN)

def tag_remap(tag: str, 
              row: Dict,
              logger: Optional[logging.Logger],
              ) -> Tags:
    """Remap tag strings to Tags enum values."""

    dict_map: Dict[str, Tags] = {
        "TRS": Tags.CONFIDENT,
        "SSP": Tags.SUSPECTED,
        "IFR": Tags.INFERRED,
        "UND": Tags.UNDECIDED
    }

    return dict_map.get(tag, 
                        Tags.UNKNOWN)

def question_types_remap(tag: str, 
                        row: Dict,
                        logger: logging.Logger,
                        endpoint: Optional[str] = "",
                        ) -> set[QuestionType]:
    """Remap tag strings to Tags enum values."""

    DEFAULT: QuestionType = QuestionType.MULTIPLE_CHOICE
    if endpoint == "" or endpoint is None:
        logger.warning("Warning: No endpoint provided for question type remapping.")
        return {DEFAULT}
    
    clean_key: str = row.get('name').replace('-', '_').lower()
    
    malf_keymap: Dict[str, str] = _malformed_keymap(keys=list(row.keys()))
    
    to_get: Any   = row.get(malf_keymap.get(f"{clean_key}_to",   ""), None)
    from_get: Any = row.get(malf_keymap.get(f"{clean_key}_from", ""), None)

    if to_get is None or from_get is None:
        logger.warning(f"Warning: Missing data for endpoint '{clean_key}': '{f"{clean_key}_to"}' or '{f"{clean_key}_from"}' not found in row.")
        return {DEFAULT}
    if int(to_get) == -99999 and int(from_get) == -99999:
        return {DEFAULT}
    
    try:
        if int(to_get) != -99999 or int(from_get) != -99999:
            if row.get("description"):
                return {QuestionType.RANGE, QuestionType.MULTIPLE_CHOICE}
            return {QuestionType.RANGE}
    except (ValueError, TypeError):
        logger.warning(f"Warning: non-integer values found for ranges {clean_key}")

    return {DEFAULT}

def _malformed_keymap(
        keys: List[str]
        ) -> Dict[str, str]:
    outs: Dict[str, str] = {}
    for k in keys:
        key: str = k.lower()
        outs[key] = k
    return outs


def collapse_prompt(messages: Dict[str, str]) -> str:
        if isinstance(messages, dict):
            messages = [messages]

        prompt: List[str] = []
        for msg in messages:
            role = msg.get("role", "").upper()
            content = msg.get("content", "").strip()
            prompt.append(f"\n\n<role>{role}</role>\n<content>{content}</content>")
        return "\n\n".join(prompt)
        

def set_seed(seed: int) -> None:
     random.seed(seed)
     np.random.seed(seed=seed)
     torch.manual_seed(seed=seed)
     if torch.cuda.is_available():
          torch.cuda.manual_seed_all(seed=seed)


def format_year(
        value: Any,
        add_end: str = True,
        ) -> str:
    """
        Format a year value, handling BCE/CE if necessary.
    """
    if not isinstance(value, int):
        raise ValueError(f"Unsupported type for year formatting: {type(value)}")
    
    end: str = ("BCE" if value < 0 else "CE")
    return f"{abs(value)} " + (end if add_end else "")

def int_round_nearest_n(
          num: float, 
          n: int = 500
          ) -> int:
     return int(round(float(num) / float(n)) * n)
    


class TqdmToLogger(io.StringIO):

    def __init__(self, logger):
        super().__init__()
        self.logger = logger
        self.ansi_scrubber = re.compile(r"\x1b\[[A-D]")

    def write(self, buf):
        cleaned = self.ansi_scrubber.sub("", buf).strip("\r\n\t ")
        if cleaned:
            self.logger.info(cleaned)

    def flush(self):
        pass