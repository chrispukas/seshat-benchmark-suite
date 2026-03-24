from enum import Enum

# ------------------------------
# ----- HYDRATION OPTIONS ------
# ------------------------------

class QuestionHydrationOptions(Enum):
    PRESENT_ABSENT = 0
    PRESENT_ABSENT_UNKNOWN = 1
    PRESENT_ABSENT_INFERREDPRESENT_INFERREDABSENT = 2
    PRESENT_ABSENT_INFERREDPRESENT_INFERREDABSENT_UNKNOWN = 3

# ---------------------
# ----- DATASETS ------
# ---------------------

class DatasetType(Enum):
    NONE = 0
    CORE = 1
    POLITY = 2
    ECONOMIC_COMPLEXITY = 3
    SOCIAL_COMPLEXITY = 4
    WARFARE_FEATURES = 5


class Tags(Enum):
    UNKNOWN = 0
    CONFIDENT = 1
    SUSPECTED = 2
    INFERRED = 3
    UNDECIDED = 4

class Quality(Enum):
    UNKNOWN = 0
    PRESENT = 1
    ABSENT = 2
    TRANSITIONAL_P_TO_A = 3
    TRANSITIONAL_A_TO_P = 4

class QuestionType(Enum):
    UNKNOWN = 0
    MULTIPLE_CHOICE = 1
    RANGE = 2


