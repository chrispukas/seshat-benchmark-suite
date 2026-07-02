from llm_benchmark.utils.utility import tag_remap, quality_remap, question_types_remap
from llm_benchmark.utils.enums import DatasetType, Tags, Quality, QuestionType, QuestionHydrationOptions

from llm_benchmark.utils.llm_interface.templates import generation_templates as gen_tmps 
from llm_benchmark.utils.utility import format_year

from typing import Dict, Any, Tuple, List, Callable

ENDPOINT_URL: str = "https://seshat-db.com/api/"
CACHE_PATH: str = "/Users/apple/Documents/github/neurips_llms/llm-bechmark/db/seshat"
DATABASE_PATH: str = "/Users/apple/Documents/github/neurips_llms/llm-bechmark/db"

GENERATION_MAXTOKENS_PER_PROMPT: int = 2048
GENERATION_TEMPERATURE: int = 0.2

EVALUATION_MAXTOKENS_PER_PROMPT: int = 2048
EVALUATION_TEMPERATURE: int = 1

BATCH_SIZE: int = 2048
SEED: int = 42

CONCURRENT_THREADS: int = 30
OPENAI_REASONING_EFFORT: str = "low"
MANUAL_REASONING: bool = True

ENABLE_API_CALLS: bool = True
BATCH_POLLING_FREQUENCY_SECONDS: int = 120

year_ranges: List[int] = [-10000, -8000, -6000, -4000, -3500, -3000, -2500, -2000, -1500, -1000, -500, 0, 500, 1000, 1500, 2000]

# -------------------------
# --- HYDRATION MAPPING ---
# -------------------------

hydrate_to_real_mapping: Dict[str, Tuple[str, Any]] = \
    { # "to_hydrate": ("real_key_in_dataset", optional_formatting_function)
        "<time-start>": ("start_year", format_year),
        "<time-end>": ("end_year", format_year),
        "<polity>": ("long_name", None),
    }


hydration_shuffle_answer_options: bool = True # Positional bias
hydration_shuffle_answer_option_labels: bool = False # Semantic Bias

hydration_answeroptions_prefix: str = "Strictly choose the correct label from the unordered set"

hydration_answeroptions: Dict[QuestionHydrationOptions, Dict[str, str]] = \
    {
        QuestionHydrationOptions.PRESENT_ABSENT: \
            {
             "A": "present", 
             "B": "absent",
             },
        QuestionHydrationOptions.PRESENT_ABSENT_UNKNOWN: \
            {
             "A": "present", 
             "B": "absent",
             "C": "unknown",
             },
        QuestionHydrationOptions.PRESENT_ABSENT_INFERREDPRESENT_INFERREDABSENT: \
            {
             "A": "present (explicit)", 
             "B": "absent  (explicit)",
             "C": "present (inferred)",
             "D": "absent  (inferred)",
             },
        QuestionHydrationOptions.PRESENT_ABSENT_INFERREDPRESENT_INFERREDABSENT_UNKNOWN: \
            {
             "A": "present (explicit)", 
             "B": "absent  (explicit)",
             "C": "present (inferred)",
             "D": "absent  (inferred)",
             "E": "unknown"
             },
    }

# --------------------------
# --- GENERATION MAPPING ---
# --------------------------


question_generation_template_mapping: Dict[QuestionType, Callable] = \
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
        "question_type_filter": ("question_type", question_types_remap)
    }


name_mapping: Dict[str, str] = \
    {
        "Gov_vio_freq_rel_grp": "Frequency_of_governmental_violence_against_religious_groups",
        "Soc_vio_freq_rel_grp": "Frequency_of_societal_violence_against_religious_groups",
        "Gov_dis_rel_grp_occ_fun": "Government_discrimination_against_religious_groups_taking_up_certain_occupations_or_functions",
        "Gov_press_conv": "Governmental_pressure_to_convert",
        "Gov_res_cir_rel_lit": "Government_restrictions_on_circulation_of_religious_literatures",
        "Gov_res_cons_rel_buil": "Government_restrictions_on_construction_of_religious_buildings",
        "Gov_res_conv": "Government_restrictions_on_conversions",
        "Gov_res_prop_own_for_rel_grp": "Government_restrictions_on_property_ownership_for_adherents_of_any_religious_groups",
        "Gov_res_pub_pros": "Government_restrictions_on_public_proselytizings",
        "Gov_res_pub_wor": "Government_restrictions_on_public_worships",
        "Gov_res_rel_edu": "Government_restrictions_on_religious_education",
        "Gov_obl_rel_grp_ofc_reco": "Government_obligations_for_religious_groups_to_apply_for_official_recognitions",
        "Soc_dis_rel_grp_occ_fun": "Societal_discrimination_against_religious_groups_taking_up_certain_occupations_or_functions",
        "Gov_press_conv_for_aga": "Societal_pressure_to_convert_or_against_conversions",
        "Sync_rel_pra_ind_beli": "Syncretism_of_religious_practices_at_the_level_of_individual_believers",
        "Tax_rel_adh_act_ins": "Taxes_based_on_religious_adherence_or_on_religious_activities_and_institutions",
        "Theo_sync_dif_rel": "Theological_syncretism_of_different_religions",
    }

unit_mapping: Dict[str, str] = \
    {
        # Social Datasets
        "sc/polity-territories": "square kilometers",
        "sc/polity-populations": "people",
        "sc/fastest-individual-communications": "days",
        "sc/largest-communication-distances": "kilometers",
        "sc/population-of-the-largest-settlements": "people",

        "sc/administrative-levels": "not applicable",
        "sc/military-levels": "not applicable",
        "sc/religious-levels": "not applicable",
        "sc/settlement-hierarchies": "not applicable"
    }


polity_mapping: Dict[str, str] = \
    {
    # Core Datasets
    "core/polities": DatasetType.POLITY,


    # Social Datasets
    "sc/research-assistants": DatasetType.SOCIAL_COMPLEXITY,
    "sc/polity-territories": DatasetType.SOCIAL_COMPLEXITY,
    "sc/polity-populations": DatasetType.SOCIAL_COMPLEXITY,
    "sc/population-of-the-largest-settlements": DatasetType.SOCIAL_COMPLEXITY,
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


    # Religious datasets
    "rt/elites-religions": DatasetType.RELIGIOUS_FEATURES,
    "rt/frequency-of-governmental-violence-against-religious-groups": DatasetType.RELIGIOUS_FEATURES,
    "rt/frequency-of-societal-violence-against-religious-groups": DatasetType.RELIGIOUS_FEATURES,
    "rt/government-discrimination-against-religious-groups-taking-up-certain-occupations-or-functions": DatasetType.RELIGIOUS_FEATURES,
    "rt/government-pressure-to-converts": DatasetType.RELIGIOUS_FEATURES,
    "rt/government-restrictions-on-circulation-of-religious-literatures": DatasetType.RELIGIOUS_FEATURES,
    "rt/government-restrictions-on-construction-of-religious-buildings": DatasetType.RELIGIOUS_FEATURES,
    "rt/government-restrictions-on-conversions": DatasetType.RELIGIOUS_FEATURES,
    "rt/government-restrictions-on-property-ownership-for-adherents-of-any-religious-groups": DatasetType.RELIGIOUS_FEATURES,
    "rt/government-restrictions-on-public-proselytizings": DatasetType.RELIGIOUS_FEATURES,
    "rt/government-restrictions-on-public-worships": DatasetType.RELIGIOUS_FEATURES,
    "rt/government-restrictions-on-religious-educations": DatasetType.RELIGIOUS_FEATURES,
    "rt/governmental-obligations-for-religious-groups-to-apply-for-official-recognitions": DatasetType.RELIGIOUS_FEATURES,
    "rt/human-sacrifices": DatasetType.RELIGIOUS_FEATURES,
    "rt/moralizing-enforcement-in-afterlife": DatasetType.RELIGIOUS_FEATURES,
    "rt/moralizing-enforcement-in-this-life": DatasetType.RELIGIOUS_FEATURES,
    "rt/moralizing-enforcement-is-agentic": DatasetType.RELIGIOUS_FEATURES,
    "rt/moralizing-enforcement-is-broad": DatasetType.RELIGIOUS_FEATURES,
    "rt/moralizing-enforcement-is-certain": DatasetType.RELIGIOUS_FEATURES,
    "rt/moralizing-enforcement-is-targeted": DatasetType.RELIGIOUS_FEATURES,
    "rt/moralizing-enforcement-of-rulers": DatasetType.RELIGIOUS_FEATURES,
    "rt/moralizing-religion-adopted-by-commoners": DatasetType.RELIGIOUS_FEATURES,
    "rt/moralizing-supernatural-concern-is-primary": DatasetType.RELIGIOUS_FEATURES,
    "rt/moralizing-religion-adopted-by-elites": DatasetType.RELIGIOUS_FEATURES,
    "rt/moralizing-supernatural-punishment-and-reward": DatasetType.RELIGIOUS_FEATURES,
    "rt/official-religions": DatasetType.RELIGIOUS_FEATURES,
    "rt/religious-fragmentations": DatasetType.RELIGIOUS_FEATURES,
    "rt/societal-discrimination-against-religious-groups-taking-up-certain-occupations-or-functions": DatasetType.RELIGIOUS_FEATURES,
    "rt/societal-pressure-to-convert-or-against-conversions": DatasetType.RELIGIOUS_FEATURES,
    "rt/syncretism-of-religious-practices-at-the-level-of-individual-believers": DatasetType.RELIGIOUS_FEATURES,
    "rt/taxes-based-on-religious-adherence-or-on-religious-activities-and-institutions": DatasetType.RELIGIOUS_FEATURES,
    "rt/theological-syncretism-of-different-religions": DatasetType.RELIGIOUS_FEATURES,
    "rt/widespread-religions": DatasetType.RELIGIOUS_FEATURES,
}