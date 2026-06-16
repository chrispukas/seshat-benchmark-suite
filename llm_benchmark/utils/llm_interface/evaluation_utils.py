import polars as pl
import random
import os

from typing import Any, Dict, List, Optional, Tuple

from llm_benchmark import config
from llm_benchmark.utils.enums import QuestionHydrationOptions
from llm_benchmark.utils.dataset import Dataset, DatasetModule

def hydrate(dataset: Dataset, 
            questions_dir: str,
            evaluation_type: QuestionHydrationOptions,
            write_path: Optional[str] = None,
            link_to_dataset: Optional[bool] = False,
            overwrite: Optional[bool] = False) -> pl.DataFrame:
    """
        Hydrate the question entries in the questions dataframe with the corresponding dataset entries, assuming datasets of one type per dataframe.
    """
    if not questions_dir or not os.path.isfile(questions_dir):
        print(f"Warning: no filepath found for the directory: {questions_dir} ")
        return None

    df: pl.DataFrame = pl.read_csv(questions_dir)
    df = df.with_columns\
        (
            pl.col("endpoint_identifier")
            .str.replace(config.ENDPOINT_URL, "")
            .str.strip_suffix("/")
        )

    unique_endpoints: List[str] = df["endpoint_identifier"].unique().to_list()

    if len(unique_endpoints) > 1:
        print(f"Warning: Multiple unique endpoints found in questions dataframe: {unique_endpoints}, failed to hydrate.")
        return None
    if not unique_endpoints:
        print("Warning: Questions dataframe is empty.")
        return None
    
    endpoint_module = dataset.get_module(unique_endpoints[0])

    if not overwrite and os.path.isfile(write_path):
        cached_df: pl.DataFrame = pl.read_csv(write_path)
        if link_to_dataset:
            endpoint_module.link_hydrated_questions(cached_df)
        return cached_df
    

    endpoint_module_df: Dict[str, pl.DataFrame] = endpoint_module.get_entries()
    hydrated_dicts: List[Dict[str, Any]] = []
 
    for row in df.iter_rows(named=True):
        entry_idx: int = int(row["entry_idx"])
        entry: pl.DataFrame = endpoint_module_df.row(entry_idx, named=True)
        
        try:
            hydrated_row: str = \
                _map_hydrated_to_real(
                to_hydrate=row["output"], 
                data=entry, 
                evaluation_type=evaluation_type
                )
            hydrated_row: str = _clean_hydrated\
                (
                    input=hydrated_row,
                )
        except KeyError:
            hydrated_row: str = ""

        hydrated_dicts.append({
                **row,
                "output": hydrated_row,
        })
        
    hydrated_df: pl.DataFrame = pl.DataFrame(hydrated_dicts)

    if write_path:
        absolute_write_path: str = os.path.abspath(write_path)
        os.makedirs(os.path.dirname(absolute_write_path), exist_ok=True)
        hydrated_df.write_csv(absolute_write_path)
        print(f"Hydrated dataframe written to {absolute_write_path}")

    if link_to_dataset:
        endpoint_module.link_hydrated_questions(hydrated_df)

    return hydrated_df


def _map_hydrated_to_real(
    to_hydrate: str,
    data: Dict[str, Any],
    evaluation_type: QuestionHydrationOptions = QuestionHydrationOptions.PRESENT_ABSENT,
    mapping: Optional[Dict[str, Tuple[str, Any]]] = config.hydrate_to_real_mapping,
) -> str:
    """Replace placeholder keys in a string with real values from the dataset."""

    if data is None:
        data: Dict[str, Any] = {}

    for key, (real_key, formatting_func) in mapping.items():
        polity: Dict[str, Any] = data.get("polity", {})
        if polity is None:
            polity: Dict[str, Any] = {}
        
        if real_key in polity:
            replace_value = polity[real_key]
            if formatting_func:
                replace_value = formatting_func(replace_value)
            to_hydrate: str = to_hydrate.replace(key, str(replace_value))
        else:
            raise KeyError(f"{real_key} not found in data['polity'].")
    if to_hydrate == "":
        return ""    
        
    answeroptions_fill: str = hydrated_answeroptions_fill(
        evaluation_type=evaluation_type,
        shuffle_options=config.hydration_shuffle_answer_options,
        shuffle_option_labels=config.hydration_shuffle_answer_option_labels
    )
    to_hydrate: str = f"{to_hydrate.split("?")[0]}? {answeroptions_fill}"
    return to_hydrate

def _clean_hydrated(
        input: str
        ) -> str:
    return input.replace('"', " ").strip()

def hydrated_answeroptions_fill(
        evaluation_type: QuestionHydrationOptions,
        shuffle_options: bool = False,
        shuffle_option_labels: bool = False,
        ) -> str:
    """
        Pseudorandom label mixup, removing bias from question label generation.
    """

    prefix: str = config.hydration_answeroptions_prefix
    options: Dict[str, str] = config.hydration_answeroptions[evaluation_type]
    option_items: List[Tuple[str, str]] = list(options.items())
    
    # Shuffle order of options if required e.g. {A = absent, B = present, C = unknown} -> {A = present, B = unknwon, C = absent}
    if shuffle_options: random.shuffle(option_items)    
    op_keys: List[str] = list(zip(*option_items))[0]

    # Shuffle Labels if required e.g. {A = absent, B = present, C = unknown} -> {A = present, B = unknwon, C = absent}
    if shuffle_option_labels: 
        _, op_values = zip(*option_items)
        op_values: List[str] = list(op_values)
        random.shuffle(op_values)
        option_items: List[Tuple[str, str]] = zip(op_keys, op_values)

    option_joined: str = ", ".join([" = ".join(pair) for pair in option_items])
    tags: List[str] = list(op_keys)
    set_string: str = ", ".join(tags)

    return f"{prefix}: {{{set_string}}}. {option_joined}"