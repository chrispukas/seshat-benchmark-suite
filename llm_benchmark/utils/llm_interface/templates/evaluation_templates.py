from typing import Dict, Any, List, Tuple

def multichoice(question: str
                         ) -> List[Dict[str, str]]:
    ROLE: str = \
        """
            You are a history expert with extensive knowledge of global history.
        """
    INSTRUCTIONS: str = \
        """
            Answer the following question to the best of your ability, ensuring that you strictly follow the question instructions, providing only the answer required.
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
            "role": "assistant",
            "content": question,
        },
    ]