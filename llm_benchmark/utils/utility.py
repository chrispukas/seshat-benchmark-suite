from typing import Dict, Optional, Any
from llm_benchmark.utils.enums import Tags, Quality, QuestionType

def quality_remap(quality: str, 
                  row: Dict
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
              row: Dict
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

def question_type_remap(tag: str, 
                        row: Dict,
                        endpoint: Optional[str] = ""
                        ) -> QuestionType:
    """Remap tag strings to Tags enum values."""

    DEFAULT: QuestionType = QuestionType.MULTIPLE_CHOICE
    print(endpoint)

    if not endpoint or endpoint == "":
        print("No endpoint provided for question type remapping.")
        return DEFAULT
    
    endpoint_to: str = f"{endpoint.replace("-", "_")[:-1]}_to"
    endpoint_from: str = f"{endpoint.replace("-", "_")[:-1]}_from"
    
    to_get: Any = row.get(endpoint_to, None)
    from_get: Any = row.get(endpoint_from, None)

    if to_get is None or from_get is None:
        print(f"Missing data for endpoint '{endpoint}': '{endpoint_to}' or '{endpoint_from}' not found in row.")
        return DEFAULT
    
    return QuestionType.RANGE