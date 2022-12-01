import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, TypeVar, Type, Any

import torch
import transformers
from cached_path import cached_path
from tango.common import det_hash

logger = logging.getLogger(__name__)


@dataclass
class TransformerSpec:
    cls: type
    model_name: str
    override_weights_file: Optional[str] = None
    override_weights_strip_prefix: Optional[str] = None
    load_weights: bool = True
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash((
            f"{self.cls.__module__}.{self.cls.__name__}",
            self.model_name,
            self.override_weights_file,
            self.override_weights_strip_prefix,
            self.load_weights,
            det_hash(self.kwargs)
        ))


_model_cache: Dict[TransformerSpec, transformers.PreTrainedModel] = {}


T = TypeVar('T')


def get(
    cls: Type[T],
    model_name: str,
    make_copy: bool,
    override_weights_file: Optional[str] = None,
    override_weights_strip_prefix: Optional[str] = None,
    load_weights: bool = True,
    **kwargs,
) -> T:
    """
    Returns a transformer model from the cache.

    # Parameters

    cls : `type`
        The type of model we are constructing.
    model_name : `str`
        The name of the transformer, for example `"bert-base-cased"`
    make_copy : `bool`
        If this is `True`, return a copy of the model instead of the cached model itself. If you want to modify the
        parameters of the model, set this to `True`. If you want only part of the model, set this to `False`, but
        make sure to `copy.deepcopy()` the bits you are keeping.
    override_weights_file : `str`, optional (default = `None`)
        If set, this specifies a file from which to load alternate weights that override the
        weights from huggingface. The file is expected to contain a PyTorch `state_dict`, created
        with `torch.save()`.
    override_weights_strip_prefix : `str`, optional (default = `None`)
        If set, strip the given prefix from the state dict when loading it.
    load_weights : `bool`, optional (default = `True`)
        If set to `False`, no weights will be loaded. This is helpful when you only
        want to initialize the architecture, like when you've already fine-tuned a model
        and are going to load the weights from a state dict elsewhere.
    """
    global _model_cache
    spec = TransformerSpec(
        cls,
        model_name,
        override_weights_file,
        override_weights_strip_prefix,
        load_weights,
        kwargs
    )
    transformer = _model_cache.get(spec, None)
    if transformer is None:
        if not load_weights:
            config = transformers.AutoConfig.from_pretrained(model_name, **kwargs)
            transformer = cls.from_config(config)   # type: ignore
        elif override_weights_file is not None:
            override_weights_file = cached_path(override_weights_file)
            override_weights = torch.load(override_weights_file)
            if override_weights_strip_prefix is not None:
                prefix = str(override_weights_strip_prefix)     # mypy insanity

                def strip_prefix(s: str) -> str:
                    if s.startswith(prefix):
                        return s[len(prefix) :]
                    else:
                        return s

                valid_keys = {
                    k
                    for k in override_weights.keys()
                    if k.startswith(prefix)
                }
                if len(valid_keys) > 0:
                    logger.info(
                        "Loading %d tensors from %s", len(valid_keys), override_weights_file
                    )
                else:
                    raise ValueError(
                        f"Specified prefix of '{prefix}' means no tensors "
                        f"will be loaded from {prefix}."
                    )
                override_weights = {strip_prefix(k): override_weights[k] for k in valid_keys}

            # load from config to avoid loading default weights
            config = transformers.AutoConfig.from_pretrained(model_name, **kwargs)
            transformer = cls.from_config(config)   # type: ignore
            # When DistributedDataParallel or DataParallel is used, the state dict of the
            # DistributedDataParallel/DataParallel wrapper prepends "module." to all parameters
            # of the actual model, since the actual model is stored within the module field.
            # This accounts for if a pretained model was saved without removing the
            # DistributedDataParallel/DataParallel wrapper.
            if hasattr(transformer, "module"):
                transformer.module.load_state_dict(override_weights)
            else:
                transformer.load_state_dict(override_weights)
        else:
            transformer = cls.from_pretrained(  # type: ignore
                model_name,
                **kwargs,
            )

        _model_cache[spec] = transformer
    if make_copy:
        import copy

        return copy.deepcopy(transformer)
    else:
        return transformer


@dataclass
class TokenizerSpec:
    cls: type
    model_name: str
    kwargs: Dict[str, Any]

    def __hash__(self):
        return hash((
            f"{self.cls.__module__}.{self.cls.__name__}",
            self.model_name,
            det_hash(self.kwargs),
        ))


_tokenizer_cache: Dict[TokenizerSpec, transformers.PreTrainedTokenizer] = {}


def get_tokenizer(cls: Type[T], model_name: str, **kwargs) -> T:
    cache_key = TokenizerSpec(cls, model_name, kwargs)

    global _tokenizer_cache
    tokenizer = _tokenizer_cache.get(cache_key, None)
    if tokenizer is None:
        # Currenty GPT2's fast tokenizer does NOT support adding a BOS token.                                                                                      
        # This issue will be fixed soon, see: https://github.com/huggingface/tokenizers/pull/1005. so that the fast tokenizer works correctly.  
        if model_name.startswith('facebook/opt'):
            kwargs['use_fast'] = False
        elif model_name.startswith('t5-'):
            # Workaround for another huggingface tokenizer bug.
            kwargs['model_max_length'] = int(1e30)
        tokenizer = cls.from_pretrained(  # type: ignore
            model_name,
            **kwargs,
        )
        _tokenizer_cache[cache_key] = tokenizer
    return tokenizer


def _clear_caches():
    """
    Clears in-memory transformer and tokenizer caches.
    """
    global _model_cache
    global _tokenizer_cache
    _model_cache.clear()
    _tokenizer_cache.clear()
