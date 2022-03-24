from catwalk2.models.gpt import GPTModel
from catwalk2.models.huggingface import HFAutoModel

MODELS = {
    "gpt2": GPTModel("gpt2"),
    "bert-base-uncased": HFAutoModel("bert-base-uncased"),
    "bert-base-cased": HFAutoModel("bert-base-cased")
}