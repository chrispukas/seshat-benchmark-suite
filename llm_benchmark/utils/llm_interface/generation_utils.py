from typing import Dict, List, Any


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