import os
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
    remapped_row["polity"] = row.get("polity_name", "Unknown")
    remapped_row["short_name"] = row.get("name", "Unknown")
    remapped_row["full_name"] = row.get("full_name", "Unknown")
    remapped_row["time_start"] = row.get(f"{endpoint}_from", "Unknown")
    remapped_row["time_end"] = row.get(f"{endpoint}_to", "Unknown")
    remapped_row["description"] = row.get("description", "Unknown")
    remapped_row["data_unit"] = row.get("data_unit", "Unknown")

    return remapped_row