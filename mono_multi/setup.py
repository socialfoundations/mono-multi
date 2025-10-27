from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier

RESULTS_ROOT_DIR = Path("./results/")
RESULTS_CSV_SAME_PROMPT = RESULTS_ROOT_DIR / "overview_results_same_prompt.csv"
RESULTS_CSV_VARY_PROMPT = RESULTS_ROOT_DIR / "overview_results_vary_prompt.csv"
FIGURES_ROOT_DIR = RESULTS_ROOT_DIR / "figures/"


ACS_TASKS = (
    "ACSIncome",
    "ACSEmployment",
    "ACSTravelTime",
    "ACSPublicCoverage",
    "ACSMobility",
    "ACSHealthInsurance",
    "ACSIncomePovertyRatio",
)

TABLESHIFT_TASKS = (
    "BRFSS_Diabetes",
    "BRFSS_Blood_Pressure",
)

SIPP_TASKS = ("SIPP",)

TASKS = ACS_TASKS + TABLESHIFT_TASKS + SIPP_TASKS
PAPER_TASKS = ACS_TASKS[:2] + TABLESHIFT_TASKS[1:] + SIPP_TASKS + ACS_TASKS[2:5]

LLM_MODELS = [
    # Google Gemma2 models
    "google/gemma-2b",
    "google/gemma-1.1-2b-it",
    "google/gemma-7b",
    "google/gemma-1.1-7b-it",
    #
    "google/gemma-2-9b",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b",
    "google/gemma-2-27b-it",
    #
    # Meta Llama3 models
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    #
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    #
    "meta-llama/Meta-Llama-3.2-1B",
    "meta-llama/Meta-Llama-3.2-1B-Instruct",
    "meta-llama/Meta-Llama-3.2-3B",
    "meta-llama/Meta-Llama-3.2-3B-Instruct",
    #
    "meta-llama/Meta-Llama-3.3-70B-Instruct",
    #
    # Mistral AI models
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mixtral-8x22B-v0.1",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "mistralai/Mistral-Small-24B-Base-2501",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    #
    # Yi models
    "01-ai/Yi-6B",
    "01-ai/Yi-6B-Chat",
    "01-ai/Yi-34B",
    "01-ai/Yi-34B-Chat",
    #
    # Qwen2 models
    # "Qwen/Qwen2-1.5B",
    # "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2-7B",
    "Qwen/Qwen2-7B-Instruct",
    "Qwen/Qwen2-72B",
    "Qwen/Qwen2-72B-Instruct",
    #
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B",
    "Qwen/Qwen2.5-72B-Instruct",
    #
    # OLMo models
    "allenai/OLMo-1B-0724-hf",
    "allenai/OLMo-1B-hf",
    "allenai/OLMo-7B-0724-hf",
    "allenai/OLMo-7B-hf",
    "allenai/OLMo-7B-Instruct-hf",
    #
    "allenai/OLMo-2-1124-7B",
    "allenai/OLMo-2-1124-7B-Instruct",
    # GPT models
    "gpt-4.1",
    "gpt-3.5-turbo-0125",
]

model_families_coarse = sorted(
    [
        "Gemma",
        "Llama",
        "Mistral",
        "OLMo",
        "Qwen",
        "Yi",
    ]
)  # "GPT",
model_families = sorted(
    [
        "Gemma 2",
        "Gemma",
        # "GPT",
        # "Llama-3.3",
        "Llama 3.2",
        "Llama 3.1",
        "Llama 3",
        "Mistral",
        "OLMo",
        "Qwen",
        "Yi",
    ]
)


developer_map = {
    "google": "Google",
    "01-ai": "01.AI",
    "meta-llama": "Meta",
    "qwen": "Alibaba",
    "allenai": "AllenAI",
    "mistralai": "Mistral AI",
    "openai": "OpenAI",
}


BASELINES = {
    "Constant": DummyClassifier(strategy="prior"),
    "LogisticRegression": LogisticRegression(),
    "GBM": HistGradientBoostingClassifier(),
    "XGBoost": XGBClassifier(),
    "NN": MLPClassifier(),
}
BASELINE_RESULTS_PATH = Path("./results/baselines")


# --------------------------------------------------
# Prompting Changes
# --------------------------------------------------
num_shots = [0, 10]

formats = ["bullet", "text", "comma"]  # , "textbullet"
connectors = ["is", "=", ":"]
granularities = ["original", "low"]
feature_orders = [
    "AGEP,COW,SCHL,MAR,OCCP,POBP,RELP,WKHP,SEX,RAC1P",  # original
    "RAC1P,WKHP,AGEP,SCHL,MAR,SEX,RELP,POBP,COW,OCCP",
    "WKHP,OCCP,RAC1P,MAR,AGEP,RELP,SCHL,POBP,COW,SEX",
    "AGEP,SCHL,OCCP,MAR,COW,WKHP,RAC1P,RELP,SEX,POBP",
    "RAC1P,SEX,WKHP,RELP,POBP,OCCP,MAR,SCHL,COW,AGEP",  # reversed
]
example_orders = [
    "0,1,2,3,4,5,6,7,8,9",  # original
    "2,6,3,5,4,7,0,1,8,9",
    "1,5,3,8,6,2,7,4,9,0",
    "7,1,9,0,3,5,4,2,6,8",
    "9,8,7,6,5,4,3,2,1,0",  # reversed
]
example_compositions = ["10,0", "7,3", "balanced", "3,7", "0,10"]
map_feature_order_to_short = dict(
    zip(feature_orders, ["default", "rand 1", "rand 2", "rand 3", " reversed"])
)
map_short_to_feature_order = {v: k for k, v in map_feature_order_to_short.items()}

map_example_order_to_short = dict(
    zip(example_orders, ["default", "rand 1", "rand 2", "rand 3", "reversed"])
)
map_short_to_example_order = {v: k for k, v in map_example_order_to_short.items()}
variations = {
    "feature_order": list(map(lambda o: map_feature_order_to_short[o], feature_orders)),
    "format": formats,
    "connector": connectors,
    "granularity": granularities,
    "example_order": list(map(lambda o: map_example_order_to_short[o], example_orders)),
    "example_composition": example_compositions,
}

variations_defaults = {
    "feature_order": variations["feature_order"][0],
    "format": "bullet",
    "connector": "is",
    "granularity": "original",
    "example_order": "default",
    "example_composition": "balanced",
}
