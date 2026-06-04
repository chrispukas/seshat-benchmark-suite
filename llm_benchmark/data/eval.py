import os

import polars as pl

from tqdm import tqdm

from typing import Dict, Tuple, Any, Optional, Set, List
from llm_benchmark.utils.dataset import Dataset, DatasetModule
from llm_benchmark.utils import benchmark
from llm_benchmark import config as cfg
from llm_benchmark.data import groupings


valid_outs: Set[str] = {"absent.", "present.", "absent", "present"}

def is_answer_valid(
        val: str
        ) -> Tuple[str, bool]:
    unique_list: Set[str] = set(val.lower().split("\n"))
    for item in unique_list:
        for valid in valid_outs:
            if valid in item:
                return valid, True
            
    return "", False

def format_answer(
        val: str
        ) -> str:
    val: str = val.strip()
    if val[-1] == ".":
        return val[:-1]
    return val

def get_actual_row(dataset_df: pl.DataFrame,  
                   entry_idx: int
                   ) -> Dict[str, Any]:
    return dataset_df.row(entry_idx, named=True)

def aggregate_entry_per_hierarchy(
        dataset: Dataset, 
        dir: str,
        model: str,
        ) -> Dict[str, Any]:
    if not os.path.exists(dir):
        raise ValueError(f"Directory {dir} does not exist!")

    outs: List[pl.DataFrame] = []

    dir_list: List[str] = os.listdir(dir)
    for item in tqdm(dir_list, desc=f"{model}"):
        origin: str = item.split("_")[0]
        endpoint: str = os.path.join(origin, item.split('.')[0].replace(f"{origin}_", ""))
        answer_path: str = os.path.join(dir, item)

        tally: pl.DataFrame = tally_answers(dataset=dataset, answers_path=answer_path)
        tally: pl.DataFrame = tally.with_columns([
            pl.lit(endpoint).alias("endpoint"),
            pl.lit(model).alias("llm_model")
            ])
        outs.append(tally)
    return pl.concat(outs)

def tally_answers(
        dataset: Dataset, 
        answers_path: str
        ) -> pl.DataFrame:
    """
        Tallies answers from a given path, returning dataframes of given structures.
    """

    if not os.path.exists(answers_path):
        raise ValueError("Path to LLM answers not specified!")
    try:
        df: pl.DataFrame = pl.read_csv(answers_path)
    except:
        raise pl.exceptions.SchemaError(f"Failed to load dataframe with path: {answers_path}")
    
    endpoint_identifier: str = df[1, 0].replace(cfg.ENDPOINT_URL, "")[:-1]
    dataset_module: DatasetModule = dataset.get_module(identifier=endpoint_identifier.replace(cfg.ENDPOINT_URL, "",))
    dataset_df: pl.DataFrame = dataset_module.get_entries()
    
    data: List[Dict[str, str]] = []

    for (question_idx, row) in enumerate(df.iter_rows(named=True)):
        answer: str = row["output"]
        clean_answer, is_valid = is_answer_valid(answer.lower())
        entry_idx: int = row["entry_idx"]

        row_actual: Dict[str, Any] = get_actual_row(dataset_df=dataset_df, entry_idx=entry_idx)
        predicted: str = classify_quality(answer=answer, is_valid=is_valid)
        actual: str = row_actual.get("polity_validity", None)

        try:
            ids: Dict[str, Any] = get_ids_from_row(dataset.grouping, row_actual, endpoint_identifier)
        except (IndexError, pl.exceptions.ColumnNotFoundError):
            print("ID ERROR")
            ids: Dict[str, Any] = {}

        entry: Dict[str, object] = {
            "seshat_entry_id": entry_idx,
            "question_entry_id": question_idx,
            "model_answer": predicted,
            "actual_answer": actual,
        }
        entry.update(ids)

        data.append(entry)
    
    if data == [] or data == None:
        print(f"Warning: polity_validity not found for {answers_path}")
        return pl.DataFrame({})
    
    df_outs: pl.DataFrame = pl.DataFrame(data=data)
    return df_outs

def classify_quality(
        answer: str, 
        is_valid: bool
        ) -> str:
    """
        Classifying question by validation check. 
        Intentionally returns strings to be compatible with dataframes.
    """
    if not is_valid:
        return "inconclusive"
    
    clean_answer: str = format_answer(answer)
    return clean_answer.split(" ")[0].lower()

def get_ids_from_row(
        groupings: groupings.Groupings, 
        row: Dict[str, object],
        endpoint: str
        ) -> Dict[str, Any]:
    """Pull grouping information from SESHAT, indexing errors are an intentional failure point."""
    polity_in_row: Dict[str, object] = row["polity"]
    polity_idx: int = polity_in_row["id"]

    polity: Dict[str, Any] = groupings.get_table_by_tag(table_name="polities", tag_truthy=polity_idx)
    region: Dict[str, Any] = groupings.get_table_by_tag(table_name="regions", tag_truthy=polity["home_seshat_region"]["id"])
    macro_region: Dict[str, Any] = groupings.get_table_by_tag(table_name="macro-regions", tag_truthy=region["mac_region"])
    
    variable_hierarchy: Dict[str, Any] = groupings.get_variable_hierarchy_by_endpoint(endpoint=endpoint)
    section_outs: Dict[str, Any] = _pull_outs_single(variable_hierarchy=variable_hierarchy, table_name="sections", section_tag="section", section_func=groupings.get_table_by_tag)
    subsection_outs: Dict[str, Any] = _pull_outs_single(variable_hierarchy=variable_hierarchy, table_name="subsections", section_tag="subsection", section_func=groupings.get_table_by_tag)

    def ground_entry(entry: Dict[str, Any], label: str) -> Dict[str, str]:
        return {
            f"{label}_idx": entry["id"],
            f"{label}_str": entry["name"],
        }
    
    outs: Dict[str, Any] = {}

    outs.update(ground_entry(region, "region"))
    outs.update(ground_entry(macro_region, "macro"))
    outs.update(section_outs)
    outs.update(subsection_outs)

    return outs

def _pull_outs_single(
        variable_hierarchy: Dict[str, Any],
        table_name: str,
        section_tag: str, 
        section_func: object
        ) -> Dict[str, Any]:
    
    try:
        section_idx: int = variable_hierarchy[section_tag]        
        section: Dict[int, Any] = section_func(table_name=table_name, tag_truthy=section_idx, tag="id")
        if section is None:
            raise KeyError("Section params not found.")

        return {
                f"{section_tag}_idx": section_idx,
                f"{section_tag}_str": section["name"]
        }   
    except (KeyError, IndexError):
        return {
                f"{section_tag}_idx": None,
                f"{section_tag}_str": None,
                }