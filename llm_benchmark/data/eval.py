import os

import polars as pl

from typing import Dict, Tuple, Any, Optional, Set, List
from llm_benchmark.utils.dataset import Dataset, DatasetModule
from llm_benchmark.utils import benchmark


def tally_directory(dataset: Dataset, 
                    dir: str
                    ) -> List[Dict[str, Any]]:
    if not os.path.exists(dir):
        raise ValueError(f"Directory {dir} does not exist!")

    outs: Dict[str, Any] = {}

    dir_list: List[str] = os.listdir(dir)
    for item in dir_list:
        origin: str = item.split("_")[0]
        endpoint: str = os.path.join(origin, item.split('.')[0].replace(f"{origin}_", ""))
        answer_path: str = os.path.join(dir, item)

        tally: Dict[str] = tally_answers(dataset=dataset, answers_path=answer_path)
        outs.update({endpoint: tally})
    return outs

def tally_answers(dataset: Dataset, 
                  answers_path: str) -> Dict[str]:
    if not os.path.exists(answers_path):
        raise ValueError("Path to LLM answers not specified!")
        return
    
    try:
        df: pl.DataFrame = pl.read_csv(answers_path)
    except:
        print(f"Failed to load dataframe with path: {answers_path}")
        return
    
    endpoint_identifier: str = df[1, 0].replace("https://seshat-db.com/api/", "")[:-1]
    dataset_df: pl.DataFrame = dataset.get_module(identifier=endpoint_identifier.replace("https://seshat-db.com/api/", "",)).get_entries()
    
    output: Dict[str, Tuple[int, List[str]]] = {
        "inconclusive":      (0, []),
        "correct_present":   (0, []),
        "correct_absent":    (0, []),
        "incorrect_present": (0, []),
        "incorrect_absent":  (0, []),
    }
    for row in df.iter_rows(named=True):
        answer: str = row["output"]
        clean_answer, is_valid = is_answer_valid(answer)
        if not is_valid:
            output["inconclusive"] = reconstruct_output(item=output["inconclusive"], to_append=answer)
            continue
        clean_answer: str = format_answer(clean_answer)
        row: Dict[str, Any] = get_actual_row(dataset_df=dataset_df, entry_idx=row["entry_idx"])
        actual: str = validate_row(row=row)

        if actual.lower() == "unknown":
            output["inconclusive"] = reconstruct_output(item=output["inconclusive"], to_append=answer)
            continue

        truth: bool = compare_answer(model=clean_answer, actual=actual)

        type = "correct" if truth else "incorrect"
        key: str = f"{type}_{clean_answer.lower()}"
        output[key] = reconstruct_output(item=output[key], to_append=answer)

    return output

def reconstruct_output(item: Tuple[int, str], to_append: str) -> Tuple[int, str]:
    count, answers_list = item
    answers_list.append(to_append)
    return (count + 1, answers_list)


def validate_row(row: Dict[str, Any]) -> str:
    
    try:
        if "polity_from" not in row.keys() or "polity_to" not in row.keys():
            raise ValueError("Ranged polity not possible")
        polity_from: int = row.get("polity_from")
        polity_to: int = row.get("polity_to")

        if polity_from is None and polity_to is None:
            return "Absent"
        
        # Specific fix for the "0" values we saw (like Peiligang)
        if polity_from == 0 and polity_to is None:
            return "Absent"
        
        return "Present"

    except:
        polity_category: str = row.get("polity_validity")
        return polity_category.lower()

def compare_answer(model: str, actual: str) -> bool:
    return model.lower() == actual.lower()

valid_outs: Set[str] = {"Absent.", "Present.", "Absent", "Present"}

def is_answer_valid(val: str) -> Tuple[str, bool]:
    val: str = val.strip();
    if val == "":
        return None, False

    first: str = val.split(" ")[0]
    return first, first in valid_outs

def format_answer(val: str) -> str:
    val: str = val.strip()
    if val[-1] == ".":
        return val[:-1]
    return val

def get_actual_row(dataset_df: pl.DataFrame,  
                   entry_idx: int
                   ) -> Dict[str, Any]:
    return dataset_df.row(entry_idx, named=True)
