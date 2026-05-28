import os
import polars as pl
from tqdm import tqdm

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
        llm_interface: LLMInterfaceModule,
        evaluation_type: QuestionHydrationOptions = QuestionHydrationOptions.PRESENT_ABSENT,
        polity_mapping: Optional[Dict[str, str]] = config.polity_mapping,

        seshat_cache_dir: Optional[str] = None,
        categories_to_evaluate: Optional[List[str]] = ["wf"],

        unhydrated_question_save_path: str = "",
        hydrated_question_save_path: str = "",
        answer_save_path: str = "",
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
    if categories_to_evaluate is None or len(categories_to_evaluate) == 0:
        raise ValueError("No polities to evaluate provided, set variable {List[str]: polities_to_evaluate}.")

    seshat_ds: dataset.Dataset = seshat_setup(seshat_cache_dir=seshat_cache_dir, polity_mapping=polity_mapping)
    
    hydrate_per_polity(dataset=seshat_ds,
                       evaluation_type=evaluation_type,
                       unhydrated_save_path=unhydrated_question_save_path,
                       hydrated_save_path=hydrated_question_save_path,
                       categories_to_evaluate=categories_to_evaluate)


    for category in categories_to_evaluate:
        _run_per_polity_evaluate(
            dataset = seshat_ds,
            llm_instance=llm_interface,
            polity = category,

            hydrated_question_dir=os.path.join(answer_save_path, f"{category}_hydrated"),
            answer_dir = os.path.join(answer_save_path, f"{category}_answers"),
        )  

def hydrate_per_polity(dataset: dataset.Dataset,
                       evaluation_type: QuestionHydrationOptions,
                    
                       unhydrated_save_path: str,
                       hydrated_save_path: str,

                       categories_to_evaluate: Optional[List[str]] = ["wf"],
                       ):
    for category in categories_to_evaluate:
        _hydrate_per_category(
            dataset=dataset,
            evaluation_type=evaluation_type,
            polity=category,

            unhydrated_question_dir=os.path.join(unhydrated_save_path, category),
            hydrated_question_dir=os.path.join(hydrated_save_path, f"{category}_hydrated")
        )


def _hydrate_per_category(dataset: dataset.Dataset,
                          evaluation_type: QuestionHydrationOptions,
                          polity: str,

                          unhydrated_question_dir: str,
                          hydrated_question_dir: str
                          ):
    subpolity_identifiers: List[str] = dataset.get_identifiers_by_parent(polity)

    for identifier in tqdm(subpolity_identifiers, desc=f"Loading hydrated questions for category {polity}: "):
        csv_question_name: str = f"{identifier.replace('/', '_')}_questions.csv"

        hydrated_df: pl.DataFrame = eutils.hydrate(
            dataset, 
            questions_dir=os.path.join(unhydrated_question_dir, csv_question_name),
            evaluation_type=evaluation_type,
            write_path=os.path.join(hydrated_question_dir, csv_question_name),
            link_to_dataset=True,
            ) 


def _run_per_polity_evaluate(
        dataset: dataset.Dataset,
        llm_instance: LLMInterfaceModule,
        polity: str, 

        hydrated_question_dir: str,
        answer_dir: str
        ) -> None:
    
    subpolity_identifiers: List[str] = dataset.get_identifiers_by_parent(polity)

    os.makedirs(hydrated_question_dir, exist_ok=True)
    os.makedirs(answer_dir, exist_ok=True)

    for identifier in tqdm(subpolity_identifiers):
        csv_answers_name: str = f"{identifier.replace('/', '_')}_answers.csv"
        
        llm_instance.respond_to_questions(
            DatasetModule = dataset.dataset_modules[identifier],
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
        dataset_module: dataset.DatasetModule = ds.dataset_modules[identifier]
        LLMInterfaceModule.generate_questions(
            DatasetModule = dataset_module,
            params = {"max_new_tokens": 512, "temperature": 0.7},
            output_path = f"{save_path}/{polity}/{identifier.replace('/', '_')}_questions.csv",
        )



def seshat_setup(seshat_cache_dir: str, 
                 polity_mapping: Dict[str, str] = config.polity_mapping,
                 force: bool = False,
                 ) -> dataset.Dataset:
    # SESHAT Intialization
    seshat_endpoint_identifiers: Dict[str, str] = seshat_requests.root_search_url(
        "https://seshat-db.com/api/", 
        use_cache=True, 
        cache_url=os.path.join(seshat_cache_dir, "cache/seshat_root_url.pkl")
    )

    seshat_module_dir: str = os.path.join(seshat_cache_dir, "main/modules")
    seshat_main_dir: str = os.path.join(seshat_cache_dir)

    os.makedirs(seshat_module_dir, exist_ok=True)
    os.makedirs(seshat_main_dir, exist_ok=True)


    seshat_ds: dataset.Dataset = dataset.Dataset(
        identifiers_endpoints=seshat_endpoint_identifiers,
        module_dir=seshat_module_dir,
        cache_dir=seshat_main_dir,
        override=force,
        ignore_polities=["crisisdb/", "core/", "general/", "rt/", "sc/", "ec/"],
        polity_mapping=polity_mapping
    )
    return seshat_ds



#seshat_ds: dataset.Dataset = seshat_setup(seshat_cache_dir="/Users/apple/Documents/github/neurips_llms/llm-bechmark/db/seshat", polity_mapping=config.polity_mapping)
    
#hydrate_per_polity(dataset=seshat_ds,
#                    evaluation_type=QuestionHydrationOptions.PRESENT_ABSENT,
#                    unhydrated_save_path="/Users/apple/Documents/github/neurips_llms/llm-bechmark/db/generation/21_04_2026/run_2_Qwen_Qwen-7B-Chat",
#                    hydrated_save_path="/Users/apple/Documents/github/neurips_llms/llm-bechmark/db/hydrated",
#                    categories_to_evaluate=['wf'])