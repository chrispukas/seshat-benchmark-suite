from typing import Dict, Any, List, Tuple, Optional

def multichoice\
    (
        question: str,
        manual_reasoning: Optional[bool] = False,
    ) -> List[Dict[str, str]]:

    ROLE: str = \
        """
            You are a history expert with extensive knowledge of global history.
        """
    SCHEMA: str = \
        """
            "Output your reasoning and answer in the following format:\n"
            "REASONING: <1-3 sentences>\n"
            "ANSWER: <letter>\n"
            "Do not include any other text."
        """ if manual_reasoning else "Answer: <selected option letter>"
        
    INSTRUCTIONS: str = \
        f"""
            You will be tasked to answer multiple-choice questions concerning the historical presence or absence of characteristics within the specified polity and time-frame, considering both direct evidence and historical inference. 
            The questions are always followed by the corresponding answer options.
            Even if you are unsure always provide an answer based on the schema, only provide the corresponding letter, and don't continue writing after giving an answer.
            {SCHEMA}
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

