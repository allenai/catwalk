from typing import Dict, Optional

from catwalk.model import Model
from catwalk.models.eleuther import EAIGPT, EAIT5
from catwalk.models.gpt import GPTModel
from catwalk.models.huggingface import HFAutoModel
from catwalk.models.rank_classification import EncoderDecoderRCModel, DecoderOnlyRCModel
from catwalk.models.t5 import T5Model, T5ModelFromPretrained
from catwalk.models.metaicl import MetaICLModel
from catwalk.models.ia3 import IA3MetaICLModel
from catwalk.models.promptsource import PromptsourceEncoderDecoderRCModel, PromptsourceDecoderOnlyRCModel
from catwalk.models.soft_prompt import with_soft_prompt

_ENCODER_DECODER_MODELS = {
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
    "bigscience/T0",
    "bigscience/T0p",
    "bigscience/T0pp",
    "bigscience/T0_single_prompt",
    "bigscience/T0_original_task_only",
    "bigscience/T0-3B",
    "google/mt5-small",
    "google/mt5-base",
    "google/mt5-large",
    "google/mt5-xl",
    "google/t5-small-lm-adapt",
    "google/t5-base-lm-adapt",
    "google/t5-large-lm-adapt",
    "google/t5-xl-lm-adapt",
    "google/t5-xxl-lm-adapt",
    "google/t5-v1_1-small",
    "google/t5-v1_1-base",
    "google/t5-v1_1-large",
    "google/t5-v1_1-xl",
    "google/t5-v1_1-xxl",
    "stas/t5-very-small-random",
    "bigscience/T0",
    "bigscience/T0p",
    "bigscience/T0pp",
    "bigscience/T0_single_prompt",
    "bigscience/T0_original_task_only",
    "bigscience/T0-3B",
    "ThomasNLG/CT0-11B"
}

_DECODER_ONLY_MODELS = {
    "sshleifer/tiny-gpt2",
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "bigscience/bloom-560m",
    "bigscience/bloom-1b1",
    "bigscience/bloom-1b7",
    "bigscience/bloom-3b",
    "bigscience/bloom-7b1",
    "bigscience/bloom",
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
    "facebook/opt-66b",
    "EleutherAI/gpt-j-6B",
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/gpt-neox-20b",
}


def _shorten_hf_name(hf_name: str) -> str:
    hf_name = hf_name.lower()
    parts = hf_name.split("/", 1)
    return parts[-1]


MODELS: Dict[str, Model] = {
    "bert-base-uncased": HFAutoModel("bert-base-uncased"),
    "bert-base-cased": HFAutoModel("bert-base-cased"),
    "bert-large-uncased": HFAutoModel("bert-large-uncased"),
    "bert-large-cased": HFAutoModel("bert-large-cased"),
    "roberta-base": HFAutoModel("roberta-base"),
    "roberta-large": HFAutoModel("roberta-large"),
    "tiny-bert": HFAutoModel("prajjwal1/bert-tiny"),
    "distilbert-base-cased-distilled-squad": HFAutoModel("distilbert-base-cased-distilled-squad"),
    "deberta-v3-base": HFAutoModel("microsoft/deberta-v3-base"),
    "deberta-v3-small": HFAutoModel("microsoft/deberta-v3-small"),
    "deberta-v3-large": HFAutoModel("microsoft/deberta-v3-large"),
    "deberta-v2-xlarge": HFAutoModel("microsoft/deberta-v2-xlarge"),
    "deberta-v2-xxlarge": HFAutoModel("microsoft/deberta-v2-xxlarge"),
}

for hf_name in _ENCODER_DECODER_MODELS:
    name = _shorten_hf_name(hf_name)
    MODELS[name] = T5ModelFromPretrained(hf_name)
    MODELS[f"eai::{name}"] = EAIT5(hf_name)
    MODELS[f"rc::{name}"] = EncoderDecoderRCModel(hf_name)
    MODELS[f"promptsource::{name}"] = PromptsourceEncoderDecoderRCModel(hf_name)

for hf_name in _DECODER_ONLY_MODELS:
    name = _shorten_hf_name(hf_name)
    MODELS[name] = GPTModel(hf_name)
    MODELS[f"eai::{name}"] = EAIGPT(hf_name)
    MODELS[f"rc::{name}"] = DecoderOnlyRCModel(hf_name)
    MODELS[f"metaicl::{name}"] = MetaICLModel(hf_name)
    MODELS[f"promptsource::{name}"] = PromptsourceDecoderOnlyRCModel(hf_name)


def short_name_for_model_object(model: Model) -> Optional[str]:
    for model_name, model_object in MODELS.items():
        if id(model) == id(model_object):
            return model_name
    return None
