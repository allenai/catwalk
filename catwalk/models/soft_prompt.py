import copy
from typing import Optional, Union

from tango.integrations.transformers import add_soft_prompt

from catwalk.model import Model


def with_soft_prompt(
    model: Union[str, Model],
    prompt_length: int,
    *,
    model_factory_method_name: str = "_make_model",
    only_prompt_is_trainable: bool = True,
    initialize_from_top_embeddings: Optional[int] = 5000,
    random_seed: int = 1940,
) -> Model:
    if isinstance(model, str):
        from catwalk.models import MODELS
        model = MODELS[model]
    model = copy.deepcopy(model)

    assert hasattr(model, model_factory_method_name)
    original_model_factory = getattr(model, model_factory_method_name)

    def new_model_factory(*args, **kwargs):
        new_model = original_model_factory(*args, **kwargs)
        add_soft_prompt(
            new_model,
            prompt_length=prompt_length,
            only_prompt_is_trainable=only_prompt_is_trainable,
            initialize_from_top_embeddings=initialize_from_top_embeddings,
            random_seed=random_seed)
        return new_model

    setattr(model, model_factory_method_name, new_model_factory)
    return model


Model.register("catwalk::with_soft_prompt")(with_soft_prompt)
