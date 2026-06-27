from typing import Any, Dict, List, Optional, Tuple

def multichoice_question(
        content: Dict[str, Any]
        ) -> List[Dict[str, str]]:
    instructions: str = f"""
    Your task is to create a challenging yet well-defined historical exam question based on a variable definition from a historical dataset.
    Follow the rules below, and return ONLY the final question template, based on the examples provided. 
    
    1. Begin the question with a concise explanation of the variable's definition, strictly adhering to the provided description, if no description is provided, use the variable name as the basis for the question.
    2. Ensure all of the following template markers are used:
        - '<polity>' for polity names (e.g. 'The Papal States')
        - '<time-start>' and '<time-end>' for temporal scope
    3. The question should make abundantly clear what the format of the desired response is and that only that response should be given.
    4. The output must be exactly one question, and must end with a question mark (?).

    Structure your response with a clear question template.
    """

    template: List[Dict[str, Any]] =  [
        {
            "role": "system",
            "content": f"""You are are history professor writing challenging questions for PhD students.\n{instructions}""",
        },
    ]

    # [(description, question), ...]
    examples: List[Dict[str, str]] = [
        {
            "name": "atlatl",
            "description": "The absence or presence of atlatl as a military technology used in warfare.",
            "question": "Consider the period from <time-start> to <time-end> in the context of '<polity>'. Was the atlatl, defined as a spear-throwing device used to increase the range and force of a projectile, absent or present as a 'military technology during this time? Strictly choose an option from the unordered set {A, B}. A = present, B = absent"
        },
        {
            "name": "ditch",
            "description": "The absence or presence of ditch as a military technology used in warfare.",
            "question": "During the period from <time-start> to <time-end>, was the use of ditches as a military technology present or absent in the warfare strategies of '<polity>'? Strictly choose an option from the unordered set {A, B}. A = present, B = absent"
        },
        {
            "name": "chain",
            "description": "The absence or presence of chainmail as a military technology used in warfare. We’re using a broad definition of chainmail. Habergeon was the word used to describe the Chinese version and that would qualify as chainmail. Armor that is made of small metal rings linked together in a pattern to form a mesh.",
            "question": "During the period from <time-start> to <time-end>, was chainmail, defined as armor made of small metal rings linked together in a pattern to form a mesh, present or absent as a military technology in the warfare practices of '<polity>'? Strictly choose an option from the unordered set {A, B}. A = present, B = absent"
        }
    ]


    examples: List[Dict[str, str]] = [item for example in examples for item in _fewshot_template_multi(example)]
    variable_name: str        = content.get("name",        "Not provided")
    variable_description: str = content.get("description", "Not provided")

    template.extend(examples)
    template.append({
                    "role": "user",
                    "content": f""""Variable name": {variable_name}\n"Variable description": {variable_description}""",
                    })
    return template

def _fewshot_template_multi(example: Dict[str, str]) -> List[Dict[str, Any]]:
    variable_name: str = example.get("name")
    description: str = example.get("description")
    question: str = example.get("question")
    
    return \
        [
            {
                "role": "user",
                "content": f"""Variable name: {variable_name}\n"Variable description: {description}"""
            },
            {
                "role": "assistant",
                "content": f"""{question}"""
            }
        ]




def range_question(content: Dict[str, Any]
                   ) -> List[Dict[str, str]]:
    instructions: str = f"""
    Your task is to create a challenging yet well-defined historical exam question based on a variable definition from a historical dataset.
    Follow the rules below, and return ONLY the final question template, based on the examples provided. 

    1. Begin the question with a concise explanation of the variable's definition, strictly adhering to the provided description, if no description is provided, use the variable name as the basis for the question.
    2. Ensure all of the following template markers are used:
        - '<polity>' for polity names (e.g. 'The Papal States')
        - '<time-start>' and '<time-end>' for temporal scope
    3. The question should make abundantly clear what the format of the desired response is and that only that response should be given.
    4. Specify the answer unit if applicable, but require that the final answer does not contain it.
    5. The answer format should be requested as: [<start_of_range>, <end_of_range>], just like a json array

    Provided variables, and description will always relate to a numeric value, so ensure that questions refer to identifying a range.
    Structure your response with a clear question template, and template markers.
    """

    template: List[Dict[str, Any]] =  [
        {
            "role": "system",
            "content": f"""You are are history professor writing challenging questions for PhD students.\n{instructions}""",
        },
    ]

    # [(description, question), ...]
    examples: List[Dict[str, str]] = \
        [
            {
                "name": "administrative level",
                "unit": "not applicable",
                "description": "Talking about Hierarchical Complexity, Administrative levels records the administrative levels of a polity. An example of hierarchy for a state society could be (1) the overall ruler, (2) provincial/regional governors, (3) district heads, (4) town mayors, (5) village heads. Note that unlike in settlement hierarchy, here you code people hierarchy. Do not simply copy settlement hierarchy data here. For archaeological polities, you will usually code as 'unknown', unless experts identified ranks of chiefs or officials independently of the settlement hierarchy. Note: Often there are more than one concurrent administrative hierarchy. In the example above the hierarchy refers to the territorial government. In addition, the ruler may have a hierarchically organized central bureaucracy located in the capital. For example, (4)the overall ruler, (3) chiefs of various ministries, (2) midlevel bureaucrats, (1) scribes and clerks. In the narrative paragraph detail what is known about both hierarchies. The machine-readable code should reflect the largest number (the longer chain of command).",
                "question": "Given the concept of Hierarchical Complexity in terms of Administrative Levels, identify the maximum number of administrative levels for the polity <polity> during the period from <time-start> to <time-end>. An example of hierarchy for a state society could be (1) the overall ruler, (2) provincial/regional governors, (3) district heads, (4) town mayors, (5) village heads. Note that unlike in settlement hierarchy, here you code people hierarchy. Provide your answer in the format [<start_of_range>, <end_of_range>].",
            },
            {
                "name": "military level",
                "unit": "not applicable",
                "description": "Talking about Hierarchical Complexity, Military levels records the Military levels of a polity. Same principle as with Administrative levels. Start with the commander-in-chief coded as: level 1, and work down to the private. Even in primitive societies such as simple chiefdoms it is often possible to distinguish at least two levels – a commander and soldiers. A complex chiefdom would be coded three levels. The presence of warrior burials might be the basis for inferring the existence of a military organization. (The lowest military level is always the individual soldier).",
                "question": "Using the definition of 'Military Level' as a measure of hierarchical complexity within a military structure, identify the range of military levels present in the polity <polity> between <time-start> and <time-end>. Your response should reflect the levels of hierarchy from the commander-in-chief to the individual soldier and should be expressed as a JSON array in the format: [<start_of_range>, <end_of_range>].",
            },
            {
                "name": "polity territory",
                "unit": "square kilometres",
                "description": "Talking about Social Scale, Polity territory is coded in squared kilometers.",
                "question": "Based on the concept of 'Polity Territory' as defined in terms of the area a polity controlled in squared kilometers, provide the range of the area <polity> governed from <time-start> to <time-end>. Please provide your answer in the format: [<start_of_range>, <end_of_range>], without the unit.",
            }
        ]
    
    examples: List[Dict[str, str]] = [item for example in examples for item in _fewshot_template_range(example)]

    variable_name: str        = content.get("name",        "Not provided").replace("_", "").lower()
    variable_description: str = content.get("description", "Not provided")
    variable_units: str       = content.get("data_unit",   None)
    
    if variable_units is None:
        raise ValueError(f"Missing 'data_unit' for entry: {content.get('name', 'Unknown name')}")

    template.extend(examples)
    template.append\
        (
            {
                "role": "user",
                "content": \
                    f"""
                        Variable name: {variable_name}
                        Variable unit: {variable_units}
                        Variable description: {variable_description}
                    """,
            }
        )
    
    return template
    
def _fewshot_template_range(example: Dict[str, str]) -> List[Dict[str, Any]]:
    variable_name: str = example.get("name")
    unit: str = example.get("unit")
    description: str = example.get("description")
    question: str = example.get("question")
    
    return \
        [
            {
                "role": "user",
                "content": \
                    f"""
                        Variable name: {variable_name}
                        Variable unit: {unit}
                        Variable description: {description}
                    """
            },
            {
                "role": "assistant",
                "content": f"""{question}"""
            }
        ]