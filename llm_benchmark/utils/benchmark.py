import os
import polars as pl

from typing import Dict, List, Optional, Any, Tuple

import llm_benchmark.config as config

import llm_benchmark.utils.dataset as dataset
import llm_benchmark.utils.seshat_requests as seshat_requests
import llm_benchmark.utils.llm_interface.evaluation_utils as eutils

from llm_benchmark.utils.enums import QuestionHydrationOptions
from llm_benchmark.utils.llm_interface.query_core import LLMInterfaceModule
from llm_benchmark.utils.llm_interface.models.local.qwen import QwenInterfaceModule

# ---------------
# -- EVALUATE ---
# ---------------

def evaluate(
        question_save_path: str,
        answer_save_path: str,

        LLMInterfaceModule: LLMInterfaceModule,
        evaluation_type: QuestionHydrationOptions = QuestionHydrationOptions.PRESENT_ABSENT,
        seshat_cache_dir: Optional[str] = None,
        polity_mapping: Optional[Dict[str, str]] = config.polity_mapping,
        polities_to_evaluate: Optional[List[str]] = ["wf"]
) -> None:
    """
        LLM evaluation function, checking LLM accuracy based on pre-hydrated questions, and corresponding dataset entries.

        Args:
            question_save_path (str): Path to source the generated questions.
            answer_save_path (str): Path to save the generated answers.
            LLMInterfaceModule (LLMInterfaceModule): An instance of the LLMInterfaceModule to use for generating answers.
            seshat_cache_dir (Optional[str]): Path for which the SESHAT database is cached to.
            polity_mapping (Optional[Dict[str, str]]): Polity mapping to use for the SESHAT dataset.
            polities_to_evaluate (Optional[List[str]]): List of polities to evaluate, if None, will evaluate the warfare features polity, tagged 'wf'.

        Returns:
            None
    """
    if seshat_cache_dir is None:
        raise ValueError("No seshat cache directory must be provided for evaluation, set variable {str: seshat_cache_dir}.")
    if polities_to_evaluate is None or len(polities_to_evaluate) == 0:
        raise ValueError("No polities to evaluate provided, set variable {List[str]: polities_to_evaluate}.")

    seshat_ds: dataset.Dataset = seshat_setup(seshat_cache_dir=seshat_cache_dir, polity_mapping=polity_mapping)

    for polity in polities_to_evaluate:
        _run_per_polity_evaluate(
            question_save_path = question_save_path,
            answer_save_path = answer_save_path,
            polity = polity,
            ds = seshat_ds,
            question_instance=LLMInterfaceModule,
            evaluation_type=evaluation_type
        )    

def _run_per_polity_evaluate(
        question_save_path: str,
        answer_save_path: str,
        polity: str, 
        ds: dataset.Dataset, 
        question_instance: LLMInterfaceModule,
        evaluation_type: QuestionHydrationOptions,
        ) -> None:
    
    subpolity_identifiers: List[str] = ds.get_identifiers_by_parent(polity)

    unhydrated_dir: str = os.path.join(question_save_path, polity)
    hydrated_dir: str = os.path.join(answer_save_path, f"{polity}_hydrated")
    answer_dir: str = os.path.join(answer_save_path, f"{polity}_answers")

    os.makedirs(hydrated_dir, exist_ok=True)
    os.makedirs(answer_dir, exist_ok=True)

    for identifier in subpolity_identifiers:
        print(f"Processing identifier: {identifier}")

        csv_questions_name: str = f"{identifier.replace('/', '_')}_questions.csv"
        csv_answers_name: str = f"{identifier.replace('/', '_')}_answers.csv"

        hydrated_df: pl.DataFrame = eutils.hydrate(ds, 
                                           questions_dir=os.path.join(unhydrated_dir, csv_questions_name),
                                           write_path=os.path.join(hydrated_dir, csv_questions_name),
                                           link_to_dataset=True,)

        question_instance.respond_to_questions(
            DatasetModule = ds.dataset_modules[identifier],
            params = {"max_new_tokens": 512, "temperature": 0.7},
            output_path = os.path.join(answer_dir, csv_answers_name),
        )

# ---------------
# -- GENERATE ---
# ---------------

def generate(
        save_path: str,
        LLMInterfaceModule: LLMInterfaceModule,
        seshat_cache_dir: Optional[str] = None,
        polity_mapping: Optional[Dict[str, str]] = config.polity_mapping,
        polities_to_evaluate: Optional[List[str]] = ["wf"]
) -> None:
    if seshat_cache_dir is None:
        raise ValueError("No seshat cache directory must be provided for evaluation, set variable {str: seshat_cache_dir}.")
    if polities_to_evaluate is None or len(polities_to_evaluate) == 0:
        raise ValueError("No polities to evaluate provided, set variable {List[str]: polities_to_evaluate}.")
    
    # SESHAT Intialization
    seshat_ds: dataset.Dataset = seshat_setup(seshat_cache_dir=seshat_cache_dir, polity_mapping=polity_mapping)

    for polity in polities_to_evaluate:
        _run_per_polity_generate(polity=polity, 
                       LLMInterfaceModule=LLMInterfaceModule, 
                       ds=seshat_ds, 
                       save_path=save_path)

def _run_per_polity_generate(polity: str, 
                    LLMInterfaceModule: LLMInterfaceModule,
                    ds: dataset.Dataset,
                    save_path: str,
                    ) -> None:
    
    polity_identifiers: List[str] = ds.get_identifiers_by_parent(polity)
    for identifier in polity_identifiers:
        print(f"Processing identifier: {identifier}")
        dataset_module: dataset.DatasetModule = ds.dataset_modules[identifier]
        LLMInterfaceModule.generate_questions(
            DatasetModule = dataset_module,
            params = {"max_new_tokens": 512, "temperature": 0.7},
            output_path = f"{save_path}/{polity}/{identifier.replace('/', '_')}_questions.csv",
        )



def seshat_setup(seshat_cache_dir: str, 
                 polity_mapping: Dict[str, str]
                 ) -> dataset.Dataset:
    # SESHAT Intialization
    seshat_endpoint_identifiers: Dict[str, str] = seshat_requests.root_search_url(
        "https://seshat-db.com/api/", 
        use_cache=True, 
        cache_url=os.path.join(seshat_cache_dir, "cache/seshat_root_url.pkl")
    )

    seshat_module_dir: str = os.path.join(seshat_cache_dir, "main/modules")
    seshat_main_dir: str = os.path.join(seshat_cache_dir, "main")

    os.makedirs(seshat_module_dir, exist_ok=True)
    os.makedirs(seshat_main_dir, exist_ok=True)


    seshat_ds: dataset.Dataset = dataset.Dataset(
        identifiers_endpoints=seshat_endpoint_identifiers,
        module_dir=seshat_module_dir,
        main_dir=seshat_main_dir,
        override=False,
        ignore_polities=["crisisdb/", "core/", "general/", "rt/"],
        polity_mapping=polity_mapping
    )
    return seshat_ds