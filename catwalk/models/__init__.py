from typing import Dict

from catwalk.model import Model
from catwalk.models.eleuther import EAIGPT, EAIT5
from catwalk.models.gpt import GPTModel
from catwalk.models.huggingface import HFAutoModel
from catwalk.models.t5 import T5Model, T5ModelFromPretrained

MODELS: Dict[str, Model] = {
    "gpt2": GPTModel("gpt2"),
    "eai::gpt2": EAIGPT("gpt2"),
    "eai::tiny-gpt2": EAIGPT("sshleifer/tiny-gpt2"),
    "eai::t5-small": EAIT5("t5-small"),
    "eai::t5-base": EAIT5("t5-base"),
    "eai::t5-large": EAIT5("t5-large"),
    "eai::t5-3b": EAIT5("t5-3b"),
    "eai::t5-11b": EAIT5("t5-11b"),
    "eai::t0": EAIT5("bigscience/T0"),
    "eai::t0p": EAIT5("bigscience/T0p"),
    "eai::t0pp": EAIT5("bigscience/T0pp"),
    "eai::t0_single_prompt": EAIT5("bigscience/T0_single_prompt"),
    "eai::t0_original_task_only": EAIT5("bigscience/T0_original_task_only"),
    "eai::t0-3b": EAIT5("bigscience/T0_3B"),
    "eai::mt5-small": EAIT5("google/mt5-small"),
    "eai::mt5-base": EAIT5("google/mt5-base"),
    "eai::mt5-large": EAIT5("google/mt5-large"),
    "eai::mt5-xl": EAIT5("google/mt5-xl"),
    "eai::t5-small-lm-adapt": EAIT5("google/t5-small-lm-adapt"),
    "eai::t5-base-lm-adapt": EAIT5("google/t5-base-lm-adapt"),
    "eai::t5-large-lm-adapt": EAIT5("google/t5-large-lm-adapt"),
    "eai::t5-xl-lm-adapt": EAIT5("google/t5-xl-lm-adapt"),
    "eai::t5-xxl-lm-adapt": EAIT5("google/t5-xxl-lm-adapt"),
    "eai::t5-very-small-random": EAIT5("stas/t5-very-small-random"),
    "bert-base-uncased": HFAutoModel("bert-base-uncased"),
    "bert-base-cased": HFAutoModel("bert-base-cased"),
    "t5-base": T5ModelFromPretrained("t5-base"),
    "t5-large": T5ModelFromPretrained("t5-large"),
    "t5-small": T5ModelFromPretrained("t5-small"),
    "t5-3b": T5ModelFromPretrained("t5-3b"),
    "t5-11b": T5ModelFromPretrained("t5-11b"),
    "t5-large-lm-adapt": T5ModelFromPretrained("google/t5-large-lm-adapt"),
    "t5-small-lm-adapt": T5ModelFromPretrained("google/t5-small-lm-adapt"),
    "t5-base-lm-adapt": T5ModelFromPretrained("google/t5-base-lm-adapt"),
    "t5-xl-lm-adapt": T5ModelFromPretrained("google/t5-xl-lm-adapt"),
    "t5-xxl-lm-adapt": T5ModelFromPretrained("google/t5-xxl-lm-adapt"),
}
