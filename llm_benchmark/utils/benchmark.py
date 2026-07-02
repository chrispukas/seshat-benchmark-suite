import os
import polars as pl
import logging

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from typing import Dict, List, Optional, Any, Tuple

import llm_benchmark.config as config

import llm_benchmark.utils.dataset as dataset
import llm_benchmark.utils.seshat_requests as seshat_requests
import llm_benchmark.utils.llm_interface.evaluation_utils as eutils

from llm_benchmark.utils.enums import QuestionHydrationOptions, QuestionType
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
        allowed_question_types: List[QuestionType] = [QuestionType.MULTIPLE_CHOICE],

        batch_size: int = config.BATCH_SIZE,

        unhydrated_question_save_path: str = "",
        hydrated_question_save_path: str = "",
        answer_save_path: str = "",

        overwrite: Optional[bool] = False,
        manual_reasoning: Optional[bool] = False,
) -> None:
    """
        LLM evaluation function, checking LLM accuracy based on pre-hydrated questions, and corresponding dataset entries.

        Args:
            llm_interface (llm_benchmark.utils.llm_interface.query_core.LLMInterfaceModule): An instance of the LLMInterfaceModule to use for generating answers.
            evaluation_type (llm_benchmark.utils.enums.QuestionHydrationOptions): The detail in which a question is asked
            polity_mapping (Optional[Dict[str, str]]): Polity mapping to use for the SESHAT dataset.

            seshat_cache_dir (Optional[str]): Path for which the SESHAT database is cached to.
            categories_to_evaluate (Optional[List[str]]): List of categories to evaluate, default behaviour is to evaluate the warfare features polity, tagged 'wf'.

            unhydrated_question_save_path (str): Path to the unhydrated questions.
            hydrated_question_save_path (str): Path to the hydrated questions.
            answer_save_path (str): Path to save the generated answers.

            overwrite (bool): Toggles if already hyrated questions are to be overwritten.
            
        Returns:
            None
    """
    if seshat_cache_dir is None:
        raise ValueError("No seshat cache directory must be provided for evaluation, set variable {str: seshat_cache_dir}.")
    if categories_to_evaluate is None or len(categories_to_evaluate) == 0:
        raise ValueError("No polities to evaluate provided, set variable {List[str]: polities_to_evaluate}.")

    seshat_ds: dataset.Dataset = seshat_setup(seshat_cache_dir=seshat_cache_dir, polity_mapping=polity_mapping)

    for category in categories_to_evaluate:
        for question_type in allowed_question_types:
            _hydrate_per_category(
                dataset=seshat_ds,
                evaluation_type=evaluation_type,
                polity=category,
                allowed_question_type=question_type,

                unhydrated_question_dir=unhydrated_question_save_path,
                hydrated_question_dir=hydrated_question_save_path,

                overwrite=overwrite,
            )

            _run_per_polity_evaluate(
                ds = seshat_ds,
                llm_instance=llm_interface,
                polity = category,
                allowed_question_type=question_type,

                hydrated_question_dir=hydrated_question_save_path,
                answer_dir=answer_save_path,
                manual_reasoning=manual_reasoning,

                batch_size=batch_size,
            )  


def _hydrate_per_category(dataset: dataset.Dataset,
                          evaluation_type: QuestionHydrationOptions,
                          polity: str,
                          allowed_question_type: QuestionType,

                          unhydrated_question_dir: str,
                          hydrated_question_dir: str,

                          overwrite: Optional[bool] = False,
                          ):
    subpolity_identifiers: List[str] = dataset.get_identifiers_by_parent(polity)

    for identifier in tqdm(subpolity_identifiers, desc=f"Loading hydrated questions for category {polity}: "):
        csv_question_name: str = f"{identifier.replace('/', '_')}_questions.csv"
        final: str = os.path.join(polity, allowed_question_type.name.lower(), csv_question_name)

        questions_dir: str = os.path.join(unhydrated_question_dir, final)

        if not questions_dir or not os.path.isfile(questions_dir):
            print(f"Warning: no filepath found for the directory: {questions_dir} ")
            continue

        df: pl.DataFrame = pl.read_csv(questions_dir)
        model: str = df.select("model").item(0, 0)

        hydrated_df: pl.DataFrame = eutils.hydrate(
            dataset, 
            questions_dir=questions_dir,
            evaluation_type=evaluation_type,
            write_path=os.path.join(hydrated_question_dir, model, final),
            link_to_dataset=True,
            overwrite=overwrite,
            ) 
        
        print(f"Hydrated {questions_dir}")


def _run_per_polity_evaluate(
        ds: dataset.Dataset,
        llm_instance: LLMInterfaceModule,
        polity: str, 
        allowed_question_type: QuestionType,

        hydrated_question_dir: str,
        answer_dir: str,
        manual_reasoning: bool,

        batch_size: int = config.BATCH_SIZE
        ) -> None:
    
    def _single(
            identifier: str
    ):
        logger: logging.Logger = logging.getLogger(identifier)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        home_dir: str = os.path.join(answer_dir, polity)
        safe_filename: str = identifier.replace("/", "_")
        log_path: str = os.path.join(home_dir, "logs", allowed_question_type.name.lower(), f"{safe_filename}_evaluation.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        file_handler: logging.FileHandler = logging.FileHandler(log_path, mode="a")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )

        logger.addHandler(file_handler)
        logger.info(f"Starting question evaluation for {identifier}.")

        try: 
            dataset_module: dataset.DatasetModule = ds.dataset_modules[identifier]
            llm_instance.respond_to_questions(
                DatasetModule=dataset_module,
                params=\
                    {
                        "max_new_tokens": config.EVALUATION_MAXTOKENS_PER_PROMPT, 
                        "temperature":    config.EVALUATION_TEMPERATURE
                    },
                output_path=os.path.join(answer_dir, polity, allowed_question_type.name.lower(), f"{identifier.replace('/', '_')}_answers.csv"), 

                logger=logger,
                allowed_question_type=allowed_question_type,
                manual_reasoning=manual_reasoning,
                batch_size=batch_size
            )
        except Exception as e:
            logger.error(f"Execution failed: {str(e)}", exc_info=True)
        finally:
            file_handler.close()
            logger.removeHandler(file_handler)
        print(f"Saved log for identifier: {identifier} in path: {log_path}")
    
    os.makedirs(hydrated_question_dir, exist_ok=True)
    os.makedirs(answer_dir, exist_ok=True)

    polity_identifiers: List[str] = ds.get_identifiers_by_parent(polity)
    with ThreadPoolExecutor(max_workers=config.CONCURRENT_THREADS) as executor:
        for t in executor._threads:
            t.daemon = True

        executor.map(_single, polity_identifiers)


# ---------------
# -- GENERATE ---
# ---------------

def generate(
        save_path: str,
        LLMInterfaceModule: LLMInterfaceModule,
        seshat_cache_dir: Optional[str] = None,
        polity_mapping: Optional[Dict[str, str]] = config.polity_mapping,
        polities_to_evaluate: Optional[List[str]] = ["wf"],
        batch_size: Optional[int] = config.BATCH_SIZE,
        allowed_question_types: List[QuestionType] = [QuestionType.MULTIPLE_CHOICE, QuestionType.RANGE]
) -> None:
    if seshat_cache_dir is None:
        raise ValueError("No seshat cache directory must be provided for evaluation, set variable {str: seshat_cache_dir}.")
    if polities_to_evaluate is None or len(polities_to_evaluate) == 0:
        raise ValueError("No polities to evaluate provided, set variable {List[str]: polities_to_evaluate}.")
    
    # SESHAT Intialization
    seshat_ds: dataset.Dataset = seshat_setup(seshat_cache_dir=seshat_cache_dir, polity_mapping=polity_mapping)

    for polity in polities_to_evaluate:
        for allowed_question_type in allowed_question_types:
            _run_per_polity_generate(
                polity=polity, 
                llm_interface=LLMInterfaceModule, 
                ds=seshat_ds, 
                save_path=save_path,
                batch_size=batch_size,
                allowed_question_type=allowed_question_type
                )

def _run_per_polity_generate(
        polity: str, 
        llm_interface: LLMInterfaceModule,
        ds: dataset.Dataset,
        save_path: str,
        allowed_question_type: QuestionType, 
        batch_size: int = config.BATCH_SIZE,
        ) -> None:
    def _single(
            identifier: str
            ):
        logger = logging.getLogger(identifier)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        home_dir: str = os.path.join(save_path, polity)

        safe_filename = identifier.replace("/", "_")
        log_path = os.path.join(home_dir, "logs", allowed_question_type.name.lower(), f"{safe_filename}_generation.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )

        logger.addHandler(file_handler)
        logger.info(f"Starting question generation for {identifier}.")

        try:
            dataset_module: dataset.DatasetModule = ds.dataset_modules[identifier]
            llm_interface.generate_questions(
                DatasetModule = dataset_module,
                params = \
                    {
                        "max_new_tokens": config.GENERATION_MAXTOKENS_PER_PROMPT, 
                        "temperature":    config.GENERATION_TEMPERATURE
                    },
                output_path = os.path.join(home_dir, allowed_question_type.name.lower(), f"{identifier.replace('/', '_')}_questions.csv"),
                batch_size=batch_size,

                logger=logger,
                allowed_question_type=allowed_question_type
            )
            logger.info(f"Successfully completed question generation for {identifier}.")
        except Exception as e:
            logger.error(f"Execution failed: {str(e)}", exc_info=True)

        finally:
            file_handler.close()
            logger.removeHandler(file_handler)
        print(f"Saved log for identifier: {identifier} in path: {log_path}")

    polity_identifiers: List[str] = ds.get_identifiers_by_parent(polity)
    with ThreadPoolExecutor(max_workers=config.CONCURRENT_THREADS) as executor:
        for t in executor._threads:
            t.daemon = True

        executor.map(_single, polity_identifiers)



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
        ignore_polities=["crisisdb/", "core/", "general/"],
        polity_mapping=polity_mapping
    )
    return seshat_ds



#seshat_ds: dataset.Dataset = seshat_setup(seshat_cache_dir="/Users/apple/Documents/github/neurips_llms/llm-bechmark/db/seshat", polity_mapping=config.polity_mapping)
    
#hydrate_per_polity(dataset=seshat_ds,
#                    evaluation_type=QuestionHydrationOptions.PRESENT_ABSENT,
#                    unhydrated_save_path="/Users/apple/Documents/github/neurips_llms/llm-bechmark/db/generation/21_04_2026/run_2_Qwen_Qwen-7B-Chat",
#                    hydrated_save_path="/Users/apple/Documents/github/neurips_llms/llm-bechmark/db/hydrated",
#                    categories_to_evaluate=['wf'])