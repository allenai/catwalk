from dataclasses import dataclass
from typing import Dict, Any, Optional
import re
from catwalk.model import TrainableModel
from catwalk.models.rank_classification import DecoderOnlyRCModel, TrainableRankClassificationModel

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_utils import Conv1D
from transformers import AutoModelForCausalLM, GPT2LMHeadModel, AutoTokenizer

from catwalk import cached_transformers
from catwalk.models import MetaICLModel

class DecoderOnlyIA3Mixin:
    @classmethod
    def _make_model(self, pretrained_model_name_or_path: str, *, ia3_weights_file: str = None, **kwargs) -> GPT2LMHeadModel:
        model = cached_transformers.get(AutoModelForCausalLM, pretrained_model_name_or_path, True)
        assert pretrained_model_name_or_path in MODEL_NAME_TO_CONFIG, f"{pretrained_model_name_or_path} not in built in IA3 configs. Please add your own IA3ForGPT2Config."
        config = MODEL_NAME_TO_CONFIG[pretrained_model_name_or_path]
        model = modify_with_ia3(model, config)
        if ia3_weights_file is not None:
            state_dict = torch.load(ia3_weights_file)
            model.load_state_dict(state_dict, strict=False)

        model.requires_grad_(False)
        for p_name, v in dict(model.named_parameters()).items():
            if re.fullmatch('.*' + config.ia3_param_names + '.*', p_name):
                v.requires_grad_(True)
        
        return model

    def trainable_copy(self) -> TrainableModel:
        return TrainableRankClassificationModel(
            self._make_model(self.pretrained_model_name_or_path),
            cached_transformers.get_tokenizer(AutoTokenizer, self.pretrained_model_name_or_path),
            self.predict_chunk
        )


class IA3DecoderOnlyRCModel(DecoderOnlyIA3Mixin, DecoderOnlyRCModel):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        *,
        likelihood_averaging: str = 'char',
        ia3_weights_file: str = None,
        **model_kwargs
    ):
        super().__init__(
            pretrained_model_name_or_path,
            likelihood_averaging=likelihood_averaging,
            ia3_weights_file=ia3_weights_file,
            **model_kwargs
        )

class IA3MetaICLModel(DecoderOnlyIA3Mixin, MetaICLModel):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        *,
        likelihood_averaging: str = 'token',
        max_length_per_example: int = 256,
        continuation_seperator: str = '\n',
        example_seperator: str = '\n\n\n',
        ia3_weights_file: str = None,
        **model_kwargs
    ):
        super().__init__(
            pretrained_model_name_or_path,
            likelihood_averaging=likelihood_averaging,
            max_length_per_example=max_length_per_example,
            continuation_seperator=continuation_seperator,
            example_seperator=example_seperator,
            ia3_weights_file=ia3_weights_file,
            **model_kwargs
        )

@dataclass
class WithIA3Config:
    attention_modules: str = None
    fused_qkv_layers: Optional[str] = None
    k_layers: Optional[str] = None
    v_layers: Optional[str] = None
    mlp_modules: str = None
    mlp_layers: str = None
    ia3_param_names: str = None

GPT_J_IA3_CONFIG = WithIA3Config(
        attention_modules = ".*attn",
        k_layers="k_proj",
        v_layers="v_proj",
        mlp_modules = ".*mlp",
        mlp_layers = "fc_in",
        ia3_param_names = "ia3"
    )

GPT_2_IA3_CONFIG = WithIA3Config(
        attention_modules = ".*attn",
        fused_qkv_layers = "c_attn",
        mlp_modules = ".*mlp",
        mlp_layers = "c_fc",
        ia3_param_names = "ia3"
    )

OPT_IA3_CONFIG = WithIA3Config(
        attention_modules = ".*self_attn",
        k_layers="k_proj",
        v_layers="v_proj",
        mlp_modules = ".*layers\.\d*",
        mlp_layers = "fc1",
        ia3_param_names = "ia3"
    )

BLOOM_IA3_CONFIG = WithIA3Config(
        attention_modules = ".*self_attention",
        fused_qkv_layers = "query_key_value",
        mlp_modules = ".*mlp",
        mlp_layers = "dense_h_to_4h",
        ia3_param_names = "ia3"
    )

MODEL_NAME_TO_CONFIG = {
    'gpt2' : GPT_2_IA3_CONFIG,
    'gpt2-medium': GPT_2_IA3_CONFIG,
    'gpt2-large': GPT_2_IA3_CONFIG,
    'gpt2-xl': GPT_2_IA3_CONFIG,
    'bigscience/bloom-560m': BLOOM_IA3_CONFIG,
    'bigscience/bloom-1b1': BLOOM_IA3_CONFIG,
    'bigscience/bloom-1b7': BLOOM_IA3_CONFIG,
    'bigscience/bloom-3b': BLOOM_IA3_CONFIG,
    'bigscience/bloom-7b1': BLOOM_IA3_CONFIG,
    'bigscience/bloom': BLOOM_IA3_CONFIG,
    'facebook/opt-125m': OPT_IA3_CONFIG,
    'facebook/opt-350m': OPT_IA3_CONFIG,
    'facebook/opt-1.3b': OPT_IA3_CONFIG,
    'facebook/opt-2.7b': OPT_IA3_CONFIG,
    'facebook/opt-6.7b': OPT_IA3_CONFIG,
    'facebook/opt-13b': OPT_IA3_CONFIG,
    'facebook/opt-30b': OPT_IA3_CONFIG,
    'facebook/opt-66b': OPT_IA3_CONFIG,
    'EleutherAI/gpt-j-6B': GPT_J_IA3_CONFIG,
}

class LinearWithIA3(nn.Module):
    def __init__(self, linear_layer, ia3_param_names, unfuse_size: int = None):
        super().__init__()

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.unfuse_size = unfuse_size

        self.weight = linear_layer.weight
        self.bias = linear_layer.bias

        self.ia3_param_names = ia3_param_names

        # if (q,k,v) are stacked into one layer
        if unfuse_size is not None:
            assert linear_layer.out_features == unfuse_size * 3
            # IA3 only operates on k and v (not q), thus the "* 2"
            setattr(self, ia3_param_names, nn.Parameter(torch.ones(unfuse_size * 2, 1)))
        else:
            setattr(self, ia3_param_names, nn.Parameter(torch.ones(self.out_features, 1)))

    def forward(self, x):
        x = F.linear(x, self.weight, self.bias)

        ia3_params = getattr(self, self.ia3_param_names)

        if ia3_params.requires_grad:
            if self.unfuse_size is not None:
                # non_q means k and v
                q, non_q = x[:, :, :self.unfuse_size], x[:, :, self.unfuse_size:]
                ia3_params = getattr(self, self.ia3_param_names)
                non_q = non_q * ia3_params.flatten()
                x = torch.cat([q, non_q], dim=2)
            else:
                x = x * ia3_params.flatten()

        return x

class Conv1DWithIA3(nn.Module):
    def __init__(self, conv1d_layer, ia3_param_names, unfuse_size: int = None):
        super().__init__()

        # nf: number of output features; nx: number of input features
        self.nf = conv1d_layer.nf
        self.unfuse_size = unfuse_size

        self.weight = conv1d_layer.weight
        self.bias = conv1d_layer.bias

        self.ia3_param_names = ia3_param_names

        # in c_att parameters, (q,k,v) linear layers are stacked into one Conv1D layer
        if unfuse_size is not None:
            assert conv1d_layer.nf == unfuse_size * 3
            # but IA3 only operates on k and v (not q), thus the "* 2"
            setattr(self, ia3_param_names, nn.Parameter(torch.ones(unfuse_size * 2, 1)))
        else:
            setattr(self, ia3_param_names, nn.Parameter(torch.ones(self.nf, 1)))

    def forward(self, x):
        # copied and pasted from the original Conv1D implemnetation
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out) # ... * self.nf

        ia3_params = getattr(self, self.ia3_param_names)

        if ia3_params.requires_grad:
            if self.unfuse_size is not None:
                # non_q means k and v
                q, non_q = x[:, :, :self.unfuse_size], x[:, :, self.unfuse_size:]
                ia3_params = getattr(self, self.ia3_param_names)
                non_q = non_q * ia3_params.flatten()
                x = torch.cat([q, non_q], dim=2)
            else:
                x = x * ia3_params.flatten()

        return x

def modify_with_ia3(transformer, config: WithIA3Config):
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(config.attention_modules, m_name) or re.fullmatch(config.mlp_modules, m_name):
            attn_layers = [regex for regex in (config.fused_qkv_layers, config.k_layers, config.v_layers) if regex is not None]
            layers_to_change = "|".join(attn_layers) \
                if re.fullmatch(config.attention_modules, m_name) \
                else config.mlp_layers
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(layers_to_change, c_name):
                    assert isinstance(layer, Conv1D) or isinstance(layer, nn.Linear), f"This code only supports Conv1D and nn.Linear"
                    adaptor_class = Conv1DWithIA3 if isinstance(layer, Conv1D) else LinearWithIA3
                    new_module = adaptor_class(
                                    layer,
                                    config.ia3_param_names,
                                    unfuse_size=transformer.config.hidden_size \
                                        if config.fused_qkv_layers and re.fullmatch(config.fused_qkv_layers, c_name) \
                                        else None
                                )
                    setattr(module, c_name, new_module)
    return transformer