from typing import Dict, Any, List, Tuple

def multichoice(question: str
                         ) -> List[Dict[str, str]]:
    ROLE: str = \
        """
            You are a history expert with extensive knowledge of global history.
        """
    INSTRUCTIONS: str = \
        """
            Answer the following questions to the best of your ability, ensuring that you strictly follow the question instructions, providing only the answer required.
        """
    
    QUESTION_1: str = \
        """
            During the period from 1747 CE to 1826 CE, was the atlatl, defined as a spear-throwing device used in warfare, present or absent in the military technology of the Americas? Please provide your answer as either 'present' or 'absent'.
        """
    ANSWER_1: str = \
        """
            present
        """

    QUESTION_2: str = \
        """
            Was the atlatl, defined as a spear-throwing device used in warfare, present or absent in the warfare practices of 'Ghur Principality' during the period from 1025 CE to 1215 CE in the New World? Please provide your answer strictly as either 'present' or 'absent'.
        """
    ANSWER_2: str = \
        """
            absent
        """
    
    QUESTION_3: str = \
        """
            During the period from 256 BCE to 125 BCE, was the atlatl, a weapon of the Americas known for its exceptional throwing distance and power, present or absent as a military technology in the warfare practices of 'Greco-Bactrian Kingdom'? Please provide your answer strictly as either 'present' or 'absent'.
        """
    ANSWER_3: str = \
        """
            absent
        """


    return [
        {
            "role": "system",
            "content": ROLE,
        },
        {
            "role": "user",
            "content": INSTRUCTIONS,
        },


        {
            "role": "user",
            "content": f"""Question description: {QUESTION_1}""".strip(),
        },
        {
            "role": "assistant",
            "content": f"""{ANSWER_1}""",
        },

        {
            "role": "user",
            "content": f"""Question description: {QUESTION_2}""".strip(),
        },
        {
            "role": "assistant",
            "content": f"""{ANSWER_2}""",
        },

        {
            "role": "user",
            "content": f"""Question description: {QUESTION_3}""".strip(),
        },
        {
            "role": "assistant",
            "content": f"""{ANSWER_3}""",
        },

        {
            "role": "user",
            "content": question,
        },
    ]