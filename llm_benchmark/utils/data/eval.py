import os

import polars as pl

from typing import Dict, Tuple, Any, Optional, Set
from llm_benchmark.utils.dataset import Dataset, DatasetModule
from llm_benchmark.utils import benchmark


pl.Config.set_tbl_rows(1000)     # increase row display
pl.Config.set_tbl_cols(100) 


def tally_answers(dataset: Dataset, 
                  answers_path: str) -> None:
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
    
    output: Dict[str] = {
        "inconclusive": 0,
        "correct_present": 0,
        "correct_absent": 0,
        "incorrect_present": 0,
        "incorrect_absent": 0,
    }

    for row in df.iter_rows(named=True):
        answer: str = row["output"]
        clean_answer, is_valid = is_answer_valid(answer)
        if not is_valid:
            output["inconclusive"] += 1;
            continue
        clean_answer: str = format_answer(clean_answer)
        row: Dict[str, Any] = get_actual_row(dataset_df=dataset_df, entry_idx=row["entry_idx"])
        truth: bool = compare_answer(model=clean_answer, actual=validate_row(row=row))

        type = "correct" if truth else "incorrect"
        output[f"{type}_{clean_answer.lower()}"] += 1

    return output

def validate_row(row: Dict[str, Any]) -> str:
    
    try:
        polity_from: int = row.get("polity_from")
        polity_to: int = row.get("polity_to")

        if polity_from is None and polity_to is None:
            return "Absent"
        
        # Specific fix for the "0" values we saw (like Peiligang)
        if polity_from == 0 and polity_to is None:
            return "Absent"
        
        return "Present"

    except:
        print("Polity category")
        polity_category: str = row.get("polity")
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


dataset: Dataset = benchmark.seshat_setup(seshat_cache_dir="/Users/apple/Documents/github/neurips_llms/llm-bechmark/db/seshat", force=False)

outs = tally_answers(dataset=dataset, answers_path="/Users/apple/Documents/github/neurips_llms/llm-bechmark/db/evaluation/22_04_2026/run_3_Qwen_Qwen-7B-Chat/wf_answers/wf_long-walls_answers.csv")

print(outs)
