from typing import Any, Dict, List, Optional, Tuple
from llm_benchmark.config import params_to_question_mapping



class QuestionGenerationModule():
    def __init__(self):
        print(f"Initialized {self.__class__.__name__}")
        pass

    def generate_questions(self, 
                           DatasetModule: Any) -> None:
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def check_if_question_in_filter(self,
                                    question: Dict[str, Any],
                                    params: Dict[str, Any]
                                    ) -> bool:
        """Check if a question matches the filter parameters."""
        for param_key, (question_key, remap) in params_to_question_mapping.items(): 
            param_item: List[Any] = params.get(param_key, [])
            question_item: Any = remap(question.get(question_key, None), question)

            if param_item and question_item not in param_item:
                return False
        return True
