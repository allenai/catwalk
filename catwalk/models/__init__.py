from catwalk.models.gpt import GPTModel
from catwalk.models.huggingface import HFAutoModel
from catwalk.models.t5 import T5Model

MODELS = {
    "gpt2": GPTModel("gpt2"),
    "bert-base-uncased": HFAutoModel("bert-base-uncased"),
    "bert-base-cased": HFAutoModel("bert-base-cased"),
    "t5-base": T5Model("t5-base"),
    "t5-large": T5Model("t5-large"),
    "t5-small": T5Model("t5-small"),
    "t5-3b": T5Model("t5-3b"),
    "t5-11b": T5Model("t5-11b"),
    "t5-large-lm-adapt": T5Model("google/t5-large-lm-adapt"),
    "t5-small-lm-adapt": T5Model("google/t5-small-lm-adapt"),
    "t5-base-lm-adapt": T5Model("google/t5-base-lm-adapt"),
    "t5-xl-lm-adapt": T5Model("google/t5-xl-lm-adapt"),
    "t5-xxl-lm-adapt": T5Model("google/t5-xxl-lm-adapt"),
}
