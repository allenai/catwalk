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
