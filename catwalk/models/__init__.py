from typing import Dict

from catwalk.model import Model
from catwalk.models.eleuther import EleutherModel, EleutherChannelModel
from catwalk.models.gpt import GPTModel
from catwalk.models.huggingface import HFAutoModel
from catwalk.models.t5 import T5Model, T5ModelFromPretrained

MODELS: Dict[str, Model] = {
    "gpt2": GPTModel("gpt2"),
    "eai::gpt2": EleutherModel("gpt2"),
    "eai::channel_gpt2": EleutherChannelModel("gpt2"),
    "eai::gptj": EleutherModel("EleutherAI/gpt-j-6B"),
    "eai::channel_gptj": EleutherChannelModel("EleutherAI/gpt-j-6B"),
    "eai::gptneox": EleutherModel("EleutherAI/gpt-neox-20b"),
    "eai::channel_gptneox": EleutherChannelModel("EleutherAI/gpt-neox-20b"),
    "eai::tiny-gpt2": EleutherModel("sshleifer/tiny-gpt2"),
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
