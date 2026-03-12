from typing import Dict, Any, List, Tuple

def multichoice(question: str
                         ) -> List[Dict[str, str]]:
    ROLE: str = \
        """
            You are a history expert with extensive knowledge of global history.
        """
    INSTRUCTIONS: str = \
        """
            Answer the following questions to the best of your ability.
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
            "content": question,
        },
    ]