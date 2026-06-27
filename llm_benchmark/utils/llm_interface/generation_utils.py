import os
import re

from datetime import datetime

from typing import Dict, List, Any


def savepath_formatting(model_name: str, default_save_path: str) -> str:
    """Format the save path for generated questions."""
    dt: str = datetime.now().strftime("%d_%m_%Y")
    default_save_path: str = os.path.join(default_save_path, dt)
    count_savepath: int = len(os.listdir(default_save_path)) + 1 if os.path.exists(default_save_path) else 1
    run_name: str = f"run_{count_savepath}_{model_name.replace('/', '_')}"
    return os.path.join(default_save_path, run_name)

def remap_question_row(row: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
    """Remap question row based on endpoint."""
    remapped_row: Dict[str, Any] = {}
    remapped_row["name"] = row.get("name", "Not provided").replace("_", " ").title()
    remapped_row["year_from"] = row.get("year_from", "Not provided")
    remapped_row["year_to"] = row.get("year_to", "Not provided")
    remapped_row["description"] = re.sub(r'§REF§.*?§REF§', '', row.get("description", "Not provided"), flags=re.DOTALL)
    remapped_row["data_unit"] = row.get("unit", "Not provided")

    return remapped_row