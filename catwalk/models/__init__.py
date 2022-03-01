from catwalk.models.model import ModelForEvaluation
from catwalk.models.hf_auto_model import HFAutoModelForEvaluation

MODELS = {
    "bert-base-uncased": HFAutoModelForEvaluation("bert-base-uncased"),
    "bert-base-cased": HFAutoModelForEvaluation("bert-base-cased"),
    "bert-large-uncased": HFAutoModelForEvaluation("bert-large-uncased"),
    "bert-large-cased": HFAutoModelForEvaluation("bert-large-cased"),
    "roberta-base": HFAutoModelForEvaluation("roberta-base"),
    "roberta-large": HFAutoModelForEvaluation("roberta-large"),
    "deepset/roberta-base-squad2": HFAutoModelForEvaluation('deepset/roberta-base-squad2')
}