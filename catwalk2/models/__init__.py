from catwalk2.models.gpt import GPTModel
from catwalk2.models.huggingface import HFAutoModel
from catwalk2.models.t5 import T5Model

MODELS = {
    "gpt2": GPTModel("gpt2"),
    "bert-base-uncased": HFAutoModel("bert-base-uncased"),
    "bert-base-cased": HFAutoModel("bert-base-cased"),
    "t5-base": T5Model("t5-base"),
    "t5-large": T5Model("t5-large"),
    "t5-small": T5Model("t5-small"),
    "t5-3b": T5Model("t5-3b"),
    "t5-11b": T5Model("t5-11b")
}