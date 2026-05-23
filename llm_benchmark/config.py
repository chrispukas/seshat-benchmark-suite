from llm_benchmark.utils.utility import tag_remap, quality_remap, question_type_remap
from llm_benchmark.utils.enums import DatasetType, Tags, Quality, QuestionType, QuestionHydrationOptions

from llm_benchmark.utils.llm_interface.templates import generation_templates as gen_tmps 

from typing import Dict, Any, Tuple


ENDPOINT_URL: str = "https://seshat-db.com/api/"


# -------------------------
# --- HYDRATION MAPPING ---
# -------------------------


def format_year(value: Any) -> str:
    """Format a year value, handling BCE/CE if necessary."""
    if isinstance(value, int):
        if value < 0:
            return f"{abs(value)} BCE"
        else:
            return f"{value} CE"
    else:
        raise ValueError(f"Unsupported type for year formatting: {type(value)}")




hydrate_to_real_mapping: Dict[str, Tuple[str, Any]] = \
    { # "to_hydrate": ("real_key_in_dataset", optional_formatting_function)
        "<time-start>": ("start_year", format_year),
        "<time-end>": ("end_year", format_year),
        "<polity>": ("long_name", None),
    }



hydrate_evaluation_type_to_template_mapping: Dict[QuestionHydrationOptions, str] = \
    {
        QuestionHydrationOptions.PRESENT_ABSENT: "Please answer strictly with either 'present' or 'absent'",
        QuestionHydrationOptions.PRESENT_ABSENT_UNKNOWN: "Please answer strictly with either 'present', 'absent', or 'unknown'",
        QuestionHydrationOptions.PRESENT_ABSENT_INFERREDPRESENT_INFERREDABSENT: "Please answer strictly with either 'present', 'absent', 'inferred present', or 'inferred absent'",
        QuestionHydrationOptions.PRESENT_ABSENT_INFERREDPRESENT_INFERREDABSENT_UNKNOWN: "Please answer strictly with either 'present', 'absent', 'inferred present', 'inferred absent', or 'unknown'",
    }

# --------------------------
# --- GENERATION MAPPING ---
# --------------------------


question_generation_template_mapping: Dict[QuestionType, object] = \
    {
        QuestionType.MULTIPLE_CHOICE: gen_tmps.multichoice_question,
        QuestionType.RANGE: gen_tmps.range_question,
    }


question_generation_params: Dict[str, Any] = \
    {
        "temperature": 0.7,
        "max_tokens": 150,
        
        "tag_filter": frozenset([Tags.CONFIDENT]),
        "quality_filter": frozenset([Quality.UNKNOWN]),
        "question_type_filter": frozenset([QuestionType.UNKNOWN, QuestionType.MULTIPLE_CHOICE]) 
    }

# ------------------------
# --- DATABASE MAPPING ---
# ------------------------

params_to_question_mapping: Dict[str, str] = \
    {
        "tag_filter": ("tag", tag_remap),
        "quality_filter": ("quality", quality_remap),
        "question_type_filter": ("question_type", question_type_remap)
    }

polity_mapping: Dict[str, str] = \
    {
    # Unique Datasets
    "core/polities": DatasetType.POLITY,


    # Social Datasets
    "sc/research-assistants": DatasetType.SOCIAL_COMPLEXITY,
    "sc/polity-territories": DatasetType.SOCIAL_COMPLEXITY,
    "sc/polity-populations": DatasetType.SOCIAL_COMPLEXITY,
    "sc/settlement-hierarchies": DatasetType.SOCIAL_COMPLEXITY,
    "sc/administrative-levels": DatasetType.SOCIAL_COMPLEXITY,
    "sc/religious-levels": DatasetType.SOCIAL_COMPLEXITY,
    "sc/military-levels": DatasetType.SOCIAL_COMPLEXITY,
    "sc/professional-military-officers": DatasetType.SOCIAL_COMPLEXITY,
    "sc/professional-soldiers": DatasetType.SOCIAL_COMPLEXITY,
    "sc/professional-priesthoods": DatasetType.SOCIAL_COMPLEXITY,
    "sc/full-time-bureaucrats": DatasetType.SOCIAL_COMPLEXITY,
    "sc/examination-systems": DatasetType.SOCIAL_COMPLEXITY,
    "sc/merit-promotions": DatasetType.SOCIAL_COMPLEXITY,
    "sc/specialized-government-buildings": DatasetType.SOCIAL_COMPLEXITY,
    "sc/formal-legal-codes": DatasetType.SOCIAL_COMPLEXITY,
    "sc/judges": DatasetType.SOCIAL_COMPLEXITY,
    "sc/courts": DatasetType.SOCIAL_COMPLEXITY,
    "sc/professional-lawyers": DatasetType.SOCIAL_COMPLEXITY,
    "sc/irrigation-systems": DatasetType.SOCIAL_COMPLEXITY,
    "sc/drinking-water-supplies": DatasetType.SOCIAL_COMPLEXITY,
    "sc/markets": DatasetType.SOCIAL_COMPLEXITY,
    "sc/food-storage-sites": DatasetType.SOCIAL_COMPLEXITY,
    "sc/roads": DatasetType.SOCIAL_COMPLEXITY,
    "sc/bridges": DatasetType.SOCIAL_COMPLEXITY,
    "sc/canals": DatasetType.SOCIAL_COMPLEXITY,
    "sc/ports": DatasetType.SOCIAL_COMPLEXITY,
    "sc/mines-or-quarries": DatasetType.SOCIAL_COMPLEXITY,
    "sc/mnemonic-devices": DatasetType.SOCIAL_COMPLEXITY,
    "sc/nonwritten-records": DatasetType.SOCIAL_COMPLEXITY,
    "sc/written-records": DatasetType.SOCIAL_COMPLEXITY,
    "sc/scripts": DatasetType.SOCIAL_COMPLEXITY,
    "sc/non-phonetic-writings": DatasetType.SOCIAL_COMPLEXITY,
    "sc/phonetic-alphabetic-writings": DatasetType.SOCIAL_COMPLEXITY,
    "sc/lists-tables-and-classifications": DatasetType.SOCIAL_COMPLEXITY,
    "sc/calendars": DatasetType.SOCIAL_COMPLEXITY,
    "sc/sacred-texts": DatasetType.SOCIAL_COMPLEXITY,
    "sc/religious-literatures": DatasetType.SOCIAL_COMPLEXITY,
    "sc/practical-literatures": DatasetType.SOCIAL_COMPLEXITY,
    "sc/histories": DatasetType.SOCIAL_COMPLEXITY,
    "sc/philosophies": DatasetType.SOCIAL_COMPLEXITY,
    "sc/scientific-literatures": DatasetType.SOCIAL_COMPLEXITY,
    "sc/fictions": DatasetType.SOCIAL_COMPLEXITY,
    "sc/articles": DatasetType.SOCIAL_COMPLEXITY,
    "sc/tokens": DatasetType.SOCIAL_COMPLEXITY,
    "sc/precious-metals": DatasetType.SOCIAL_COMPLEXITY,
    "sc/foreign-coins": DatasetType.SOCIAL_COMPLEXITY,
    "sc/indigenous-coins": DatasetType.SOCIAL_COMPLEXITY,
    "sc/paper-currencies": DatasetType.SOCIAL_COMPLEXITY,
    "sc/couriers": DatasetType.SOCIAL_COMPLEXITY,
    "sc/postal-stations": DatasetType.SOCIAL_COMPLEXITY,
    "sc/general-postal-services": DatasetType.SOCIAL_COMPLEXITY,
    "sc/communal-buildings": DatasetType.SOCIAL_COMPLEXITY,
    "sc/utilitarian-public-buildings": DatasetType.SOCIAL_COMPLEXITY,
    "sc/symbolic-buildings": DatasetType.SOCIAL_COMPLEXITY,
    "sc/entertainment-buildings": DatasetType.SOCIAL_COMPLEXITY,
    "sc/knowledge-or-information-buildings": DatasetType.SOCIAL_COMPLEXITY,
    "sc/other-utilitarian-public-buildings": DatasetType.SOCIAL_COMPLEXITY,
    "sc/special-purpose-sites": DatasetType.SOCIAL_COMPLEXITY,
    "sc/ceremonial-sites": DatasetType.SOCIAL_COMPLEXITY,
    "sc/burial-sites": DatasetType.SOCIAL_COMPLEXITY,
    "sc/trading-emporia": DatasetType.SOCIAL_COMPLEXITY,
    "sc/enclosures": DatasetType.SOCIAL_COMPLEXITY,
    "sc/length-measurement-systems": DatasetType.SOCIAL_COMPLEXITY,
    "sc/area-measurement-systems": DatasetType.SOCIAL_COMPLEXITY,
    "sc/volume-measurement-systems": DatasetType.SOCIAL_COMPLEXITY,
    "sc/weight-measurement-systems": DatasetType.SOCIAL_COMPLEXITY,
    "sc/time-measurement-systems": DatasetType.SOCIAL_COMPLEXITY,
    "sc/geometrical-measurement-systems": DatasetType.SOCIAL_COMPLEXITY,
    "sc/other-measurement-systems": DatasetType.SOCIAL_COMPLEXITY,
    "sc/debt-and-credit-structures": DatasetType.SOCIAL_COMPLEXITY,
    "sc/stores-of-wealth": DatasetType.SOCIAL_COMPLEXITY,
    "sc/sources-of-support": DatasetType.SOCIAL_COMPLEXITY,
    "sc/occupational-complexities": DatasetType.SOCIAL_COMPLEXITY,
    "sc/special-purpose-houses": DatasetType.SOCIAL_COMPLEXITY,
    "sc/other-special-purpose-sites": DatasetType.SOCIAL_COMPLEXITY,
    "sc/largest-communication-distances": DatasetType.SOCIAL_COMPLEXITY,
    "sc/fastest-individual-communications": DatasetType.SOCIAL_COMPLEXITY,


    # Warfare datasets
    "wf/long-walls": DatasetType.WARFARE_FEATURES,
    "wf/coppers": DatasetType.WARFARE_FEATURES,
    "wf/bronzes": DatasetType.WARFARE_FEATURES,
    "wf/irons": DatasetType.WARFARE_FEATURES,
    "wf/steels": DatasetType.WARFARE_FEATURES,
    "wf/javelins": DatasetType.WARFARE_FEATURES,
    "wf/atlatls": DatasetType.WARFARE_FEATURES,
    "wf/slings": DatasetType.WARFARE_FEATURES,
    "wf/self-bows": DatasetType.WARFARE_FEATURES,
    "wf/composite-bows": DatasetType.WARFARE_FEATURES,
    "wf/crossbows": DatasetType.WARFARE_FEATURES,
    "wf/tension-siege-engines": DatasetType.WARFARE_FEATURES,
    "wf/sling-siege-engines": DatasetType.WARFARE_FEATURES,
    "wf/gunpowder-siege-artilleries": DatasetType.WARFARE_FEATURES,
    "wf/handheld-firearms": DatasetType.WARFARE_FEATURES,
    "wf/war-clubs": DatasetType.WARFARE_FEATURES,
    "wf/battle-axes": DatasetType.WARFARE_FEATURES,
    "wf/daggers": DatasetType.WARFARE_FEATURES,
    "wf/swords": DatasetType.WARFARE_FEATURES,
    "wf/spears": DatasetType.WARFARE_FEATURES,
    "wf/polearms": DatasetType.WARFARE_FEATURES,
    "wf/dogs": DatasetType.WARFARE_FEATURES,
    "wf/donkeys": DatasetType.WARFARE_FEATURES,
    "wf/horses": DatasetType.WARFARE_FEATURES,
    "wf/camels": DatasetType.WARFARE_FEATURES,
    "wf/elephants": DatasetType.WARFARE_FEATURES,
    "wf/wood-bark-etc": DatasetType.WARFARE_FEATURES,
    "wf/leathers": DatasetType.WARFARE_FEATURES,
    "wf/shields": DatasetType.WARFARE_FEATURES,
    "wf/helmets": DatasetType.WARFARE_FEATURES,
    "wf/breastplates": DatasetType.WARFARE_FEATURES,
    "wf/limb-protections": DatasetType.WARFARE_FEATURES,
    "wf/scaled-armors": DatasetType.WARFARE_FEATURES,
    "wf/laminar-armors": DatasetType.WARFARE_FEATURES,
    "wf/plate-armors": DatasetType.WARFARE_FEATURES,
    "wf/small-vessel-canoe-etc": DatasetType.WARFARE_FEATURES,
    "wf/merchant-ship-pressed-into-service": DatasetType.WARFARE_FEATURES,
    "wf/specialized-military-vessels": DatasetType.WARFARE_FEATURES,
    "wf/settlement-in-defensive-positions": DatasetType.WARFARE_FEATURES,
    "wf/wooden-palisades": DatasetType.WARFARE_FEATURES,
    "wf/earth-ramparts": DatasetType.WARFARE_FEATURES,
    "wf/ditches": DatasetType.WARFARE_FEATURES,
    "wf/moats": DatasetType.WARFARE_FEATURES,
    "wf/stone-walls-non-mortared": DatasetType.WARFARE_FEATURES,
    "wf/stone-walls-mortared": DatasetType.WARFARE_FEATURES,
    "wf/fortified-camps": DatasetType.WARFARE_FEATURES,
    "wf/complex-fortifications": DatasetType.WARFARE_FEATURES,
    "wf/modern-fortifications": DatasetType.WARFARE_FEATURES,
    "wf/chainmails": DatasetType.WARFARE_FEATURES,


    # Economic datasets
    "ec/luxury-precious-metals": DatasetType.ECONOMIC_COMPLEXITY,
    "ec/luxury-fabrics": DatasetType.ECONOMIC_COMPLEXITY,
    "ec/luxury-manufactured-goods": DatasetType.ECONOMIC_COMPLEXITY,
    "ec/luxury-spices-incense-and-dyes": DatasetType.ECONOMIC_COMPLEXITY,
    "ec/luxury-drink-alcohol": DatasetType.ECONOMIC_COMPLEXITY,
    "ec/luxury-glass-goods": DatasetType.ECONOMIC_COMPLEXITY,
    "ec/luxury-fine-ceramic-wares": DatasetType.ECONOMIC_COMPLEXITY,
    "ec/luxury-precious-stones": DatasetType.ECONOMIC_COMPLEXITY,
    "ec/luxury-statuary": DatasetType.ECONOMIC_COMPLEXITY,
    "ec/luxury-food": DatasetType.ECONOMIC_COMPLEXITY,
    "ec/other-luxury-personal-items": DatasetType.ECONOMIC_COMPLEXITY,
}