import polars as pl
import random
from typing import Any, Dict, List, Optional, Tuple

from llm_benchmark import config
from llm_benchmark.utils.enums import QuestionHydrationOptions
from llm_benchmark.utils.dataset import Dataset, DatasetModule


# --------------------
# --- METRIC UTILS ---
# --------------------


# -----------------------
# --- HYDRATION UTILS ---
# -----------------------

def hydrate(Dataset: Dataset, 
            questions_dir: str,
            EvaluationType: QuestionHydrationOptions,
            write_path: Optional[str] = None,
            link_to_dataset: Optional[bool] = False) -> pl.DataFrame:
    """Hydrate the question entries in the questions dataframe with the corresponding dataset entries, assumes datasets of one type per dataframe."""
    df: pl.DataFrame = pl.read_csv(questions_dir)
    df = df.with_columns(
                        pl.col("endpoint_identifier")
                                .str.replace("https://seshat-db.com/api/", "")
                                .str.strip_suffix("/")
                        )

    unique_endpoints: List[str] = df["endpoint_identifier"].unique().to_list()
    if len(unique_endpoints) > 1:
        print(f"Multiple unique endpoints found in questions dataframe: {unique_endpoints}, failed to hydrate.")
        return pl.DataFrame({})
    endpoint_module = Dataset.get_module(unique_endpoints[0])
    endpoint_module_df: Dict[str, pl.DataFrame] = {unique_endpoints[0]: endpoint_module.get_entries()}

    hydrated_dicts: List[Dict[str, Any]] = []

    for row in df.iter_rows(named=True):
        endpoint_identifier: str = row["endpoint_identifier"]
        entry_idx: int = int(row["entry_idx"])
        entry: str = row["output"]

        module_df: pl.DataFrame = endpoint_module_df[endpoint_identifier]
        entry: pl.DataFrame = module_df.row(entry_idx, named=True)
        hydrated_row: str = map_hydrated_to_real(to_hydrate=row["output"], data=entry, evaluation_type=EvaluationType)

        hydrated_dicts.append({
                **row,
                "output": hydrated_row,
        })
        
    hydrated_df =  pl.DataFrame(hydrated_dicts)
    if write_path:
        hydrated_df.write_csv(write_path)
        print(f"Hydrated dataframe written to {write_path}")
    if link_to_dataset:
        endpoint_module.link_hydrated_questions(hydrated_df)
    return hydrated_df


def map_hydrated_to_real(
    to_hydrate: str,
    data: Dict[str, Any],
    evaluation_type: QuestionHydrationOptions = QuestionHydrationOptions.PRESENT_ABSENT,
    mapping: Optional[Dict[str, Tuple[str, Any]]] = config.hydrate_to_real_mapping,
) -> str:
    """Replace placeholder keys in a string with real values from the dataset."""

    for key, (real_key, formatting_func) in mapping.items():
        if real_key in data["polity"]:
            replace_value = data["polity"][real_key]
            if formatting_func:
                replace_value = formatting_func(replace_value)
            to_hydrate = to_hydrate.replace(key, str(replace_value))
        else:
            raise KeyError(f"{real_key} not found in data['polity'].")
        
    answeroptions_fill: str = hydrated_answeroptions_fill(
        evaluation_type=evaluation_type,
        shuffle_options=config.hydration_shuffle_answer_options,
        shuffle_option_labels=config.hydration_shuffle_answer_option_labels
    )
    
    return to_hydrate.replace("<answer-options>", answeroptions_fill)

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
        random.shuffle(op_values)
        option_items: List[Tuple[str, str]] = zip(op_keys, op_values)

    option_joined: str = "\n".join([" = ".join(pair) for pair in option_items])
    tags: List[str] = list(op_keys)
    set_string: str = ", ".join(tags)

    return f"{prefix}: {{{set_string}}}.\n{option_joined}"