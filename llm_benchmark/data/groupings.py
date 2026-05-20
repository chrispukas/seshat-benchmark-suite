import os
import polars as pl

from typing import List, Dict, Any



class Groupings():
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        
        core_dir: str = os.path.join(self.cache_dir, "main/modules/core")

        self.variable_heiararchy: pl.DataFrame = self.load_parquet(os.path.join(core_dir, "variable-hierarchies.parquet"))
        self.sections_parquet: pl.DataFrame  = self.load_parquet(os.path.join(core_dir, "sections.parquet"))
        self.subsections_parquet: pl.DataFrame = self.load_parquet(os.path.join(core_dir, "subsections.parquet"))

        self.sections_by_id: Dict[int, Dict[str, Any]] = self.map_section_by_id(self.sections_parquet)
        self.subsections_by_id: Dict[int, Dict[str, Any]] = self.map_section_by_id(self.subsections_parquet)

        self.groupings: Dict[str, PolityGroup] = self.refresh_groupings()
    

    def get_polity_group_by_identifier(self, identifier: str) -> PolityGroup:
        return self.groupings.get(identifier, None)

    def load_parquet(self, parquet_path: str) -> pl.DataFrame:
        try:
            df = pl.read_parquet(parquet_path)
            return df
        except Exception as e:
            print(f"Error loading parquet file at {parquet_path}: {e}")
            raise e

    def refresh_groupings(self) -> Dict[str, PolityGroup]:
        groupings: Dict[str, PolityGroup] = {}
        
        for row in self.variable_heiararchy.iter_rows(named=True):
            identifier: str = row.get("api_endpoint")
            if identifier is None or identifier == "":
                print(f"   Warning: missing 'api_endpoint' in row: {row.get("name")}")
                continue

            identifier = identifier.replace("/api/", "")[0:-1]

            section_id: int = row.get("section", None)
            subsection_id: int = row.get("subsection", None)

            section_row: Dict[str, Any] = self.sections_by_id.get(section_id, {})
            subsection_row: Dict[str, Any] = self.subsections_by_id.get(subsection_id, {})

            groupings[identifier] = PolityGroup(
                identifier=identifier,
                
                seshat_section_entry_id=section_id,
                section_name=section_row.get("name", ""),

                subsection_id=subsection_id,
                subsection_name=subsection_row.get("name", "")
            )
        
        return groupings
    
    def map_section_by_id(self, df: pl.DataFrame) -> pl.DataFrame:
        sections_by_id: Dict[int, Dict[str, Any]] = {}

        for row in df.iter_rows(named=True):
            idx: int = row.get("id", None)
            row: Dict[str, Any] = row
            sections_by_id.update({idx: row})

        return sections_by_id

class PolityGroup():
    def __init__(self, 
                 identifier: str,

                 seshat_section_entry_id: int = None,
                 section_name: str = "",

                 subsection_id: int = None,
                 subsection_name: str = ""):
        
        self.identifier = identifier

        self.seshat_section_entry_id = seshat_section_entry_id
        self.section_name = section_name

        self.subsection_id: int = subsection_id
        self.subsection_name: str = subsection_name




gr = Groupings(cache_dir="/Users/apple/Documents/github/neurips_llms/llm-bechmark/db/seshat")
