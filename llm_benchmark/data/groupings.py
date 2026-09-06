import os
import polars as pl

from typing import List, Dict, Any, Tuple, Optional



class Groupings():
    """Class to manage groupings of polity variables based on Seshat's variable hierarchy."""
    def __init__(self, 
                 cache_dir: str):
        self.cache_dir = cache_dir
        
        core_dir: str = os.path.join(self.cache_dir, "main/modules/core")
        general_dir: str = os.path.join(self.cache_dir, "main/modules/general")

        parquets: List[str] = [
            "variable-hierarchies.parquet",
            "polities.parquet",
            "macro-regions.parquet",
            "regions.parquet",
            "sections.parquet",
            "subsections.parquet",
        ]

        self.tables: Dict[str, pl.DataFrame] = self.table_setup(dir=core_dir, parquet_list=parquets)

        #self.groupings: Dict[str, PolityGroup] = self.refresh_groupings()

    def table_setup(self,
                    dir: str,
                    parquet_list: str) -> Dict[str, pl.DataFrame]: 
        tables: Optional[Dict[str, pl.DataFrame]] = {}

        for pq in parquet_list:
            df: pl.DataFrame = self.load_parquet(os.path.join(dir, pq))
            tables[pq.replace(".parquet", "")] = df

        return tables
        
    def get_table_by_tag(self, 
                         table_name: str,
                         tag_truthy: Any,
                         tag: Optional[str] = "id",
                         silent: bool = True,) -> Dict[str, Any]:
        table: pl.DataFrame = self.tables.get(table_name, {})
        if table is None:
            print("Table not found!")
            return {}
        if tag_truthy == None:
            return {}
        
        rows: List[Dict[str, Any]] = table.filter(pl.col(tag) == tag_truthy).to_dicts()
        len_fetchrows: int = len(rows)
        if len_fetchrows > 1:
            if not silent: print(f"Warning: {len_fetchrows} entries for tag: {tag} in table: {table_name} for {tag} {tag_truthy}, returning first.")
        elif len_fetchrows == 0:
            if not silent: print(f"Warning: No entries for tag: {tag} in table: {table_name} for {tag} {tag_truthy}.")
            return {}
        return rows[0] or {}

    def get_variable_hierarchy_by_endpoint(self, 
                             endpoint: str) -> str:
        """Get the variable heiarchy for a given endpoint, as found in SESHAT."""
        seshat_endpoint: str = endpoint
        if not seshat_endpoint.startswith("/api/"):
            seshat_endpoint: str = f"/api/{endpoint}/"
        return self.get_table_by_tag("variable-hierarchies", tag_truthy=seshat_endpoint, tag="api_endpoint")

    def load_parquet(self, parquet_path: str) -> pl.DataFrame:
        """Load a parquet file and return a Polars DataFrame."""
        try:
            df = pl.read_parquet(parquet_path)
            return df
        except Exception as e:
            print(f"Error loading parquet file at {parquet_path}: {e}")
            raise e