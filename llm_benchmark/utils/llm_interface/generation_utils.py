from typing import Dict, List, Any



def range_message(prompt: dict) -> list[dict[str, str]]:
    RANGE_INSTRUCTIONS = """
    Your task is to create a challenging yet well-defined historical exam question based on a variable definition.
    Follow the rules below, and return ONLY the final question template. Do not include explanations, roles, metadata, or reasoning.

    1. Start with a concise explanation of the variable's definition, strictly adhering to the provided description
    2. Variables will always relate to a range, for example the range of area a polity controlled in a certain span of time
    3. Formulate a question template where <polity> marks the location for polity names (e.g., "Papal States")
    4. Include temporal scope markers: <time-start> and <time-end>
    5. These temporal scope markers should be for the general temporal range of the polity
    6. The answer format should be requested as: [<start_of_range>, <end_of_range>], just like a json array
    7. Specify the answer unit if applicable, but the answer should not contain it!
    The question should make abundantly clear what the format of the desired response is and that only that response should be given

    Structure your response with clear question template and metadata fields.
    """
    ADMIN_LEVEL_EXPLANATION = "Talking about Hierarchical Complexity, Administrative levels records the administrative levels of a polity. An example of hierarchy for a state society could be (1) the overall ruler, (2) provincial/regional governors, (3) district heads, (4) town mayors, (5) village heads. Note that unlike in settlement hierarchy, here you code people hierarchy. Do not simply copy settlement hierarchy data here. For archaeological polities, you will usually code as 'unknown', unless experts identified ranks of chiefs or officials independently of the settlement hierarchy. Note: Often there are more than one concurrent administrative hierarchy. In the example above the hierarchy refers to the territorial government. In addition, the ruler may have a hierarchically organized central bureaucracy located in the capital. For example, (4)the overall ruler, (3) chiefs of various ministries, (2) midlevel bureaucrats, (1) scribes and clerks. In the narrative paragraph detail what is known about both hierarchies. The machine-readable code should reflect the largest number (the longer chain of command)."
    MIL_LEVEL_EXPLANATION = "Talking about Hierarchical Complexity, Military levels records the Military levels of a polity. Same principle as with Administrative levels. Start with the commander-in-chief coded as: level 1, and work down to the private. Even in primitive societies such as simple chiefdoms it is often possible to distinguish at least two levels – a commander and soldiers. A complex chiefdom would be coded three levels. The presence of warrior burials might be the basis for inferring the existence of a military organization. (The lowest military level is always the individual soldier)."
    POL_TERRITORY_EXPLANATION = (
        "Talking about Social Scale, Polity territory is coded in squared kilometers."
    )

    ADMIN_LEVEL = "Given the concept of Hierarchical Complexity in terms of Administrative Levels, identify the maximum number of administrative levels for the polity <polity> during the period from <time-start> to <time-end>. An example of hierarchy for a state society could be (1) the overall ruler, (2) provincial/regional governors, (3) district heads, (4) town mayors, (5) village heads. Note that unlike in settlement hierarchy, here you code people hierarchy. Provide your answer in the format [<start_of_range>, <end_of_range>]."
    MIL_LEVEL = "Using the definition of 'Military Level' as a measure of hierarchical complexity within a military structure, identify the range of military levels present in the polity <polity> between <time-start> and <time-end>. Your response should reflect the levels of hierarchy from the commander-in-chief to the individual soldier and should be expressed as a JSON array in the format: [<start_of_range>, <end_of_range>]."
    POL_TERRITORY = "Based on the concept of 'Polity Territory' as defined in terms of the area a polity controlled in squared kilometers, provide the range of the area <polity> governed from <time-start> to <time-end>. Please provide your answer in the format: [<start_of_range>, <end_of_range>], without the unit."
    # SUPRA_INTERACTION_SCALE = "Given the variable Polity Scale Of Supracultural Interaction, which estimates the area in km squared encompassed by the supracultural entity, determine the range of this area for the <polity> between <time-start> and <time-end>. Please provide your answer in the format [<start_of_range>, <end_of_range>] without including 'km squared'."
    return [
        {
            "role": "system",
            "content": "You are are history professor writing challenging questions for PhD students.",
        },
        {
            "role": "user",
            "content": RANGE_INSTRUCTIONS,
        },
        {
            "role": "user",
            "content": f"""
         Variable name: Administrative Level
         Variable unit: not applicable
         Variable description: {ADMIN_LEVEL_EXPLANATION}""",
        },
        {
            "role": "assistant",
            "content": ADMIN_LEVEL,
        },
        {
            "role": "user",
            "content": f"""
         Variable name: Military Level
         Variable unit: not applicable
         Variable description: {MIL_LEVEL_EXPLANATION}""",
        },
        {
            "role": "assistant",
            "content": MIL_LEVEL,
        },
        {
            "role": "user",
            "content": f"""
         Variable name: Polity Territory
         Variable unit: square kilometres
         Variable description: {POL_TERRITORY_EXPLANATION}""",
        },
        {
            "role": "assistant",
            "content": POL_TERRITORY,
        },
        prompt,
    ]



def multichoice_question(params: Dict[str, Any]) -> List[Dict[str, str]]:
    ABS_PRES_INSTRUCTIONS = """
    Your task is to create a challenging yet well-defined historical exam question based on a variable definition. Follow these requirements:
    Follow the rules below, and return ONLY the final question template. Do not include explanations, roles, metadata, or reasoning.
    
    1. Start with a concise explanation of the variable's definition, strictly adhering to the provided description
    2. Formulate a question template where <polity> marks the location for polity names (e.g., "the Papal States")
    3. Include temporal scope markers: <time-start> and <time-end>
    4. The only allowed answers can be either exactly "absent" or exactly "present".
    The question should make abundantly clear what the format of the desired response is and that only that response should be given

    Structure your response with clear question template.
    """
    
    ATLATL_EXPLANATION = (
        "The absence or presence of atlatl as a military technology used in warfare."
    )
    ATLATL = "Consider the period from <time-start> to <time-end> in the context of '<polity>'. Was the atlatl, defined as a spear-throwing device used to increase the range and force of a projectile, absent or present as a 'military technology during this time? The answer should be strictly 'absent' or 'present'."

    DITCH = "During the period from <time-start> to <time-end>, was the use of ditches as a military technology present or absent in the warfare strategies of '<polity>'? Respond with either 'present' or 'absent'."
    DITCH_EXPLANATION = (
        "The absence or presence of ditch as a military technology used in warfare."
    )
    CHAIN = "During the period from <time-start> to <time-end>, was chainmail, defined as armor made of small metal rings linked together in a pattern to form a mesh, present or absent as a military technology in the warfare practices of '<polity>'? Provide your answer strictly as either 'present' or 'absent'."
    CHAIN_EXPLANATION = "The absence or presence of chainmail as a military technology used in warfare. We’re using a broad definition of chainmail. Habergeon was the word used to describe the Chinese version and that would qualify as chainmail. Armor that is made of small metal rings linked together in a pattern to form a mesh."

    return [
        {
            "role": "system",
            "content": "You are are history professor writing challenging questions for PhD students.",
        },
        {
            "role": "user",
            "content": ABS_PRES_INSTRUCTIONS,
        },
        {
            "role": "user",
            "content": f"""
        Variable name: atlatl
        Variable description: {ATLATL_EXPLANATION}""",
        },
        {
            "role": "system",
            "content": ATLATL,
        },
        {
            "role": "user",
            "content": f"""
        Variable name: ditch
        Variable description: {DITCH_EXPLANATION}""",
        },
        {
            "role": "system",
            "content": DITCH,
        },
        {
            "role": "user",
            "content": f"""
        Variable name: chain
        Variable description: {CHAIN_EXPLANATION}""",
        },
        {
            "role": "system",
            "content": CHAIN,
        },
        {
            "role": "user",
            "content": f"""
                "variable name": {params.get("full_name", "Unknown Variable")},
                "variable description": {params.get("description", "No description available.")},""",
        },
    ]
    pass

def range_question(params: Dict[str, Any]) -> List[Dict[str, str]]:
    
    pass




def remap_question_row(row: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
    """Remap question row based on endpoint."""
    remapped_row: Dict[str, Any] = {}
    remapped_row["polity"] = row.get("polity_name", "Unknown Polity")
    remapped_row["short_name"] = row.get("name", "Unknown Short Name")
    remapped_row["full_name"] = row.get("full_name", "Unknown Full Name")
    remapped_row["time_start"] = row.get(f"{endpoint}_from", "Unknown Start Time")
    remapped_row["time_end"] = row.get(f"{endpoint}_to", "Unknown End Time")
    remapped_row["description"] = row.get("description", "No description available.")
    remapped_row["data_unit"] = row.get("data_unit", "Not applicable.")

    return remapped_row