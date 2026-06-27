import os
import json

import numpy as np
import polars as pl

import llm_benchmark.utils.seshat_requests as seshat_requests
import llm_benchmark.data.groupings as groupings
import llm_benchmark.config as config

from typing import Any, Dict, List, Optional, Tuple, Set
from llm_benchmark.utils.enums import DatasetType


class Dataset():
    def __init__(self,
                 identifiers_endpoints: Dict[str, str],
                 module_dir: str,
                 cache_dir: str,
                 override: bool = False,
                 cache_endpoints: Optional[bool] = True,
                 ignore_polities: Optional[List[str]] = None,
                 polity_mapping: Optional[Dict[str, DatasetType]] = None,
                 ) -> None:

        self.identifers = list(identifiers_endpoints.keys())
        self.endpoints = list(identifiers_endpoints.values())
        
        self.endpoints_by_parent, self.parent_identifiers = self.parent_identifier_mapping()

        self.module_dir = module_dir
        self.cache_dir = cache_dir
        self.dataset_modules: Dict[str, DatasetModule] = {}

        self.override = override


        if not polity_mapping:
            polity_mapping = {}
        self.polity_mapping = polity_mapping

        if not ignore_polities:
            ignore_polities = []
        self.ignore_polities = ignore_polities


        self.grouping = groupings.Groupings(cache_dir=self.cache_dir)

        self.refresh(override=self.override, 
                     polity_mapping=self.polity_mapping)
        
    def parent_identifier_mapping(self
                                  ) -> Tuple[Dict[str, List[str]], Set[str]]:
        """Create a mapping of parent identifiers to their corresponding full identifiers."""
        mapping: Dict[str, List[str]] = {}
        parent_identifiers: Set[str] = set()
        for identifier in self.identifers:
            parent_id = identifier.split("/")[0]
            parent_identifiers.add(parent_id)
            mapping[parent_id] = mapping.get(parent_id, []) + [identifier]
        return mapping, parent_identifiers
        
    def stratify_by_type(self,
                         dataset_type: DatasetType
                         ) -> Any:
        
        module_mapping: Dict[DatasetType, Any] = {
            DatasetType.NONE: DatasetModule,
            DatasetType.CORE: DatasetModule,
            DatasetType.POLITY: PolityModule,
            DatasetType.ECONOMIC_COMPLEXITY: EconomicComplexityModule,
            DatasetType.SOCIAL_COMPLEXITY: SocialComplexityModule,
            DatasetType.WARFARE_FEATURES: WarfareFeaturesModule,
            DatasetType.RELIGIOUS_FEATURES: ReligiousFeaturesModule,
        }

        return module_mapping.get(dataset_type, DatasetModule)
    
    def refresh(self, 
                override: bool = False,
                polity_mapping: Optional[Dict[str, DatasetType]] = None) -> None:
        print("Attempting dataset refresh.", override)

        self.categorized_endpoints: Dict[DatasetType, List[str]] = {}
        self.dataset_modules: Dict[str, DatasetModule] = {}
        
        for identifier, endpoint in zip(self.identifers, self.endpoints):
            print(identifier, endpoint)
            parquet_path: str = os.path.join(self.module_dir, f"{identifier}.parquet")
            os.makedirs(os.path.dirname(parquet_path), exist_ok=True)

            if self.check_if_identifier_is_ignored(identifier):
                print(f"Ignoring polity {identifier} as per configuration.")
                continue

            dataset_type: DatasetType = polity_mapping.get(identifier, DatasetType.NONE)
            DatasetClass = self.stratify_by_type(dataset_type)
            self.dataset_modules[identifier] = DatasetClass(parquet_path=parquet_path,
                                                            seshat_identifier=identifier,
                                                            seshat_url=endpoint,
                                                            polity_group=None,
                                                            override=override)
            self.categorized_endpoints[dataset_type] = self.categorized_endpoints.get(dataset_type, []) + [identifier]
            
        print(f"Dataset refresh complete. Loaded {len(self.dataset_modules)} modules.")
        print(f"Modules: {list(self.dataset_modules.keys())}")
            
    def check_if_identifier_is_ignored(self,
                                        identifier: str
                                        ) -> bool:
        """Check if the identifier should be ignored based on the ignore_polities list."""
        for ignore_polity in self.ignore_polities or []:
            if ignore_polity in identifier:
                return True
        return False
    
    def get_module_identifiers(self) -> List[str]:
        return list(self.dataset_modules.keys())
    
    def get_categorized_endpoints(self, 
                                  dataset_type: DatasetType
                                  ) -> List[str]:
        return self.categorized_endpoints[dataset_type]
    
    def get_identifiers_by_parent(self, parent_identifier: str) -> Dict[str, List[str]]:
        res: List[str] = self.endpoints_by_parent.get(parent_identifier, [])
        if not res:
            print(f"No identifiers found for parent identifier {parent_identifier}. Available parent identifiers: {self.parent_identifiers}")
        return res

    def get_module(self, identifier: str) -> Optional[DatasetModule]:
        return self.dataset_modules.get(identifier, None)


class DatasetModule():
    def __init__(self, 
                 parquet_path: str,
                 polity_group: Optional[groupings.PolityGroup] = None,
                 dataset_type: DatasetType = DatasetType.NONE,
                 seshat_identifier: Optional[str] = None,
                 seshat_url: Optional[str] = None,
                 override: bool = False,
                 ) -> None:
        
        self.parquet_path = parquet_path
        self.seshat_identifier = seshat_identifier
        self.seshat_url = seshat_url
        self.dataset_type = dataset_type
        self.questions = None

        self.polity_group = polity_group

        if os.path.isfile(parquet_path) and not override:
            self.dataset: pl.DataFrame = pl.read_parquet(parquet_path)
        elif seshat_url is not None:
            self.refresh()
        else:
            print(ValueError("Either parquet_path must exist or seshat_url must be provided with override=True."))
            return
        print(f"Loaded module {self.seshat_identifier} with {self.dataset.height} entries.")

    def refresh(self) -> None:
        results: List[Dict[str, Any]]  = seshat_requests.traverse_polity(self.seshat_url)
        results_collapsed: List[Dict[str, Any]] = self.format_all_entries(results)
        try:
            self.dataset: pl.DataFrame = \
                pl.DataFrame(results_collapsed,
                             infer_schema_length=None)
        except Exception as e:
            print("Error during dataset refresh:", e)
            print("Sample of results causing error:", results_collapsed[0:20])
            raise e
        
        try:
            self.dataset: pl.DataFrame = self.dataset.sort("id")
        except Exception as e:
            print("Error during sorting dataset:", e)

        try:
            identifier_shorthand: str = results[0]["name"].lower()
            self.dataset: pl.DataFrame = self.dataset.rename({f"{identifier_shorthand}_from": "polity_from"})
            self.dataset: pl.DataFrame = self.dataset.rename({f"{identifier_shorthand}_to": "polity_to"})
        except:
            try:
                self.dataset: pl.DataFrame = self.dataset.rename({f"{identifier_shorthand}": "polity_validity"})
            except:
                print(f"Warning: failed to rename polity: polity_validity")
            print(f"Warning: failed to rename polity: -> polity_from. \n\n")

        self.dataset.write_parquet(self.parquet_path)
        print(f"Dataset refreshed with {self.dataset.height} entries.")

    def format_all_entries(self,
                           entries: List[Dict[str, Any]],
                           ) -> None:
        for i, entry in enumerate(entries):
            collapsed_entry: Dict[str, Any] = self.format_entry(entry)
            entries[i] = collapsed_entry
        return entries

    def format_entry(self,
                     entry: Dict[str, Any],
                     remappable_keys: Optional[Dict[str, Tuple[Any, Any]]] = None,
                     ) -> Dict[str, Any]:
        for key, value in entry.items():
            if isinstance(value, (dict, list)):
                entry[key] = json.dumps(value)

        if remappable_keys is None:
            remappable_keys: Dict[str, Tuple[Any, Any]] = {
                "name": (None, ""),
                "comment": (None, ""),
                "description": (None, ""),
            }

        for key, (old_value, new_value) in remappable_keys.items():
            if key in entry and entry[key] == old_value:
                entry[key] = new_value
        
        return entry
    
    def get_entries(self) -> pl.DataFrame:
        return self.dataset
    def get_endpoint(self) -> str:
        return self.seshat_url
    def get_questions(self) -> pl.DataFrame:
        return self.questions
    def get_seshat_identifier(self) -> str:
        return self.seshat_identifier
    def get_polity_groupings(self) -> str:
        return self.polity_group
    
    def get_polity_by_id(self,
                         polity_id: int
                         ) -> Optional[pl.DataFrame]:
        filtered_df: pl.DataFrame = self.dataset.filter(pl.col("id") == polity_id)
        if filtered_df.height > 0:
            return filtered_df
        else:
            return None

    def sanitize_row(self,
                    row: Dict[str, Any]
                    ) -> Dict[str, Any]:
        return row

    def generate_questions(self, 
                           QuestionGenerator: Any) -> None:
        QuestionGenerator.generate_questions(self)


        pass
    def link_hydrated_questions(self, 
                                df: pl.DataFrame
                            ) -> None:
        """Link the hydrated questions back to the dataset for evaluation."""
        self.questions = df

# -------------------------------
# ---- MODULES FOR EACH TYPE ----
# -------------------------------

class PolityModule(DatasetModule):
    def __init__(self, 
                 parquet_path: str,
                 dataset_type: DatasetType = DatasetType.NONE,
                 polity_group: Optional[groupings.PolityGroup] = None,
                 seshat_identifier: Optional[str] = None,
                 seshat_url: Optional[str] = None,
                 override: bool = False,
                 ) -> None:
        super().__init__(parquet_path, polity_group, DatasetType.POLITY, seshat_identifier, seshat_url, override)

    def format_entry(self,
                     entry: Dict[str, Any]
                     ) -> Dict[str, Any]:
        
        unreliable_instability_events: Optional[int] = entry.get("unreliable_instability_events", -1)
        is_empty_on_polaris_release: Optional[int] = entry.get("is_empty_on_polaris_release", -1)

        entry["unreliable_instability_events"] = unreliable_instability_events
        entry["is_empty_on_polaris_release"] = is_empty_on_polaris_release
        
        remappable_keys: Dict[str, Tuple[Any, Any]] = {
            "name": (None, ""),
            "start_year": (None, int(-99999)),
            "end_year": (None, int(-99999)),
            "long_name": (None, ""),
            "polity_tag": (None, ""),
            "general_description": (None, ""),
            "shapefile_name": (None, ""),
            "unreliable_instability_events": (None, -1),
            "is_empty_on_polaris_release": (None, -1),
            "home_nga": (None, {}),
            "home_seshat_region": (None, {}),
        }

        return super().format_entry(entry, remappable_keys)

class EconomicComplexityModule(DatasetModule):
    def __init__(self, 
                 parquet_path: str,
                 polity_group: Optional[groupings.PolityGroup] = None,
                 dataset_type: DatasetType = DatasetType.ECONOMIC_COMPLEXITY,
                 seshat_identifier: Optional[str] = None,
                 seshat_url: Optional[str] = None,
                 override: bool = False,
                 ) -> None:
        super().__init__(parquet_path=parquet_path, 
                         dataset_type=DatasetType.ECONOMIC_COMPLEXITY, 
                         seshat_identifier=seshat_identifier, 
                         seshat_url=seshat_url, 
                         override=override,
                         polity_group=polity_group)
                         
        
        print("Initialized EconomicComplexityModule.")
        

    def format_entry(self,
                     entry: Dict[str, Any]
                     ) -> Dict[str, Any]:
        
        remappable_keys: Dict[str, Tuple[Any, Any]] = {
            "year_from": (None, int(-99999)),
            "year_to": (None, int(-99999)),
            
            "tag": (None, ""),
            "coded_value": (None, ""),

            "place_of_provenance": (None, []),
            "place_of_provenance_str": (None, ""),

            "ruler_consumption": (None, ""),
            "ruler_consumption_tag": (None, ""),

            "elite_consumption": (None, ""),
            "elite_consumption_tag": (None, ""),

            "common_people_consumption": (None, ""),
            "common_people_consumption_tag": (None, ""),

            "name": (None, ""),
            "comment": (None, ""),
            "description": (None, ""),
        }

        return super().format_entry(entry, remappable_keys)
    
    def sanitize_row(self, 
                     row: Dict[str, Any]
                     ) -> Dict[str, Any]:
        parent_sanitized: Dict[str, Any] = super().sanitize_row(row)
        sanitized: Dict[str, Any] = {
            "type": self.dataset_type,

            "name": parent_sanitized.get("name", ""),
            "comment": parent_sanitized.get("comment", ""),
            "description": parent_sanitized.get("description", ""),

            "date_to": parent_sanitized.get("year_to", -99999),
            "date_from": parent_sanitized.get("year_from", -99999),
        }

        return sanitized



class SocialComplexityModule(DatasetModule):
    def __init__(self, 
                 parquet_path: str,
                 dataset_type: DatasetType = DatasetType.SOCIAL_COMPLEXITY,
                 seshat_identifier: Optional[str] = None,
                 seshat_url: Optional[str] = None,
                 override: bool = False,
                 polity_group: Optional[groupings.PolityGroup] = None
                 ) -> None:
        self.unit = config.unit_mapping.get(seshat_identifier, "")
        super().__init__(parquet_path=parquet_path, 
                         dataset_type=DatasetType.SOCIAL_COMPLEXITY, 
                         seshat_identifier=seshat_identifier, 
                         seshat_url=seshat_url, 
                         override=override,
                         polity_group=polity_group)
        
        print("Initialized SocialComplexityModule.")
        

    def format_entry(self,
                     entry: Dict[str, Any]
                     ) -> Dict[str, Any]:
        

        name: str = entry.get("name", "")
        is_disputed: Optional[int] = entry.get("is_disputed", -1)
        is_uncertain: Optional[int] = entry.get("is_uncertain", -1)

        entry["is_disputed"] = is_disputed
        entry["is_uncertain"] = is_uncertain
        entry["unit"] = self.unit
        
        remappable_keys: Dict[str, Tuple[Any, Any]] = {
            "year_from": (None, int(-99999)),
            "year_to": (None, int(-99999)),
            
            "tag": (None, ""),
            "coded_value": (None, ""),

            "is_disputed": (None, -1),
            "is_uncertain": (None, -1),

            "name": (None, ""),

            f"{name.lower()}_from": (None, -99999),
            f"{name.lower()}_to": (None, -99999),

            "comment": (None, ""),
            "description": (None, ""),

            "unit": (None, "")

        }

        return super().format_entry(entry, remappable_keys)
    
class WarfareFeaturesModule(DatasetModule):
    def __init__(self, 
                 parquet_path: str,
                 dataset_type: DatasetType = DatasetType.WARFARE_FEATURES,
                 seshat_identifier: Optional[str] = None,
                 seshat_url: Optional[str] = None,
                 override: bool = False,
                 polity_group: Optional[groupings.PolityGroup] = None
                 ) -> None:
        super().__init__(parquet_path=parquet_path, 
                         dataset_type=DatasetType.WARFARE_FEATURES, 
                         seshat_identifier=seshat_identifier, 
                         seshat_url=seshat_url, 
                         override=override,
                         polity_group=polity_group)
        
        print("Initialized WarfareFeaturesModule.")
        

    def format_entry(self,
                     entry: Dict[str, Any]
                     ) -> Dict[str, Any]:
        

        name: str = entry.get("name", "")
        is_disputed: Optional[int] = entry.get("is_disputed", -1)
        is_uncertain: Optional[int] = entry.get("is_uncertain", -1)

        entry["is_disputed"] = is_disputed
        entry["is_uncertain"] = is_uncertain
        
        remappable_keys: Dict[str, Tuple[Any, Any]] = {
            "year_from": (None, int(-99999)),
            "year_to": (None, int(-99999)),
            
            "tag": (None, ""),
            "coded_value": (None, ""),

            "is_disputed": (None, -1),
            "is_uncertain": (None, -1),

            "name": (None, ""),

            f"{name.lower()}": (None, ""),

            "comment": (None, ""),
            "description": (None, ""),

        }

        return super().format_entry(entry, remappable_keys)
    
class ReligiousFeaturesModule(DatasetModule):
    def __init__(self, 
                 parquet_path: str,
                 dataset_type: DatasetType = DatasetType.RELIGIOUS_FEATURES,
                 seshat_identifier: Optional[str] = None,
                 seshat_url: Optional[str] = None,
                 override: bool = False,
                 polity_group: Optional[groupings.PolityGroup] = None
                 ) -> None:
        super().__init__(parquet_path=parquet_path, 
                         dataset_type=DatasetType.RELIGIOUS_FEATURES, 
                         seshat_identifier=seshat_identifier, 
                         seshat_url=seshat_url, 
                         override=override,
                         polity_group=polity_group)
        
        print("Initialized WarfareFeaturesModule.")
        

    def format_entry(self,
                     entry: Dict[str, Any]
                     ) -> Dict[str, Any]:
        

        name: str = entry.get("name", "")
        entry["name"] = config.name_mapping.get(name, name)
        is_disputed: Optional[int] = entry.get("is_disputed", -1)
        is_uncertain: Optional[int] = entry.get("is_uncertain", -1)

        entry["is_disputed"] = is_disputed
        entry["is_uncertain"] = is_uncertain
        
        remappable_keys: Dict[str, Tuple[Any, Any]] = {
            "year_from": (None, int(-99999)),
            "year_to": (None, int(-99999)),
            
            "tag": (None, ""),
            "coded_value": (None, ""),

            "is_disputed": (None, -1),
            "is_uncertain": (None, -1),

            "name": (None, ""),

            f"{name.lower()}": (None, ""),

            "comment": (None, ""),
            "description": (None, ""),

        }

        return super().format_entry(entry, remappable_keys)


