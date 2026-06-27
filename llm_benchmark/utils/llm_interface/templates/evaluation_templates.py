from typing import Dict, Any, List, Tuple

def multichoice(question: str
                         ) -> List[Dict[str, str]]:
    ROLE: str = \
        """
            You are a history expert with extensive knowledge of global history.
        """
    INSTRUCTIONS: str = \
        """
            You will be tasked to answer multiple-choice questions concerning the historical presence or absence of characteristics within the specified polity and time-frame, considering both direct evidence and historical inference. 
            Provide a concise reasoning, then answer the questions. 
            The questions are always followed by the corresponding answer options.
            Even if you are unsure always provide an answer based on the schema, only provide the corresponding letter, and don't continue writing after giving an answer.
        
            Follow the output schema:
            Answer: <selected option letter>
            
            Do not output anything after this line.
        """


    return [
        {
            "role": "system",
            "content": f"""{ROLE}\n{INSTRUCTIONS}""",
        },
        {
            "role": "user",
            "content": question,
        },
    ]

