import re
from typing import Optional

import torch
import torch.nn as nn
from transformers.modeling_utils import Conv1D
from transformers import AutoModelForCausalLM, GPT2LMHeadModel

from catwalk import cached_transformers
from catwalk.models import MetaICLModel


class DecoderOnlyIA3Mixin:
    @classmethod
    def _make_model(
        self,
        pretrained_model_name_or_path: str,
        *,
        ia3_weights_file: Optional[str] = None,
        **kwargs
    ) -> GPT2LMHeadModel:
        model = cached_transformers.get(AutoModelForCausalLM, pretrained_model_name_or_path, True)
        isinstance(model, GPT2LMHeadModel)
        config = IA3ForGPT2Config()
        model = modify_with_ia3(model, config)
        state_dict = torch.load(ia3_weights_file)
        model.load_state_dict(state_dict, strict=False)
        return model


class IA3MetaICLModel(DecoderOnlyIA3Mixin, MetaICLModel):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        *,
        likelihood_averaging: str = 'token',
        max_length_per_example: int = 256,
        continuation_seperator: str = '\n',
        example_seperator: str = '\n\n\n',
        ia3_weights_file: Optional[str] = None,
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
        assert ia3_weights_file is not None


# The following code comes from from allenai/hn-icl by Qinyuan Yu, used with permission

class IA3ForGPT2Config:
    def __init__(self):
        self.att_modules = ".*attn"
        self.att_layers = "c_attn"
        self.mlp_modules = ".*mlp"
        self.mlp_layers = "c_fc"
        self.trainable_param_names = ".*lora_b.*"


class Conv1DAttWithIA3(nn.Module):
    def __init__(self, conv1d_layer, hidden_size):
        super().__init__()

        # nf: number of output features; nx: number of input features
        self.nf = conv1d_layer.nf
        self.hidden_size = hidden_size

        # in c_att parameters, (q,k,v) linear layers are stacked into one Conv1D layer
        assert conv1d_layer.nf == hidden_size * 3

        self.weight = conv1d_layer.weight
        self.bias = conv1d_layer.bias

        # but IA3 only operats on k and v (no q), thus the "* 2"
        self.multi_lora_b = nn.Parameter(torch.ones(hidden_size * 2, 1))

    def forward(self, x):
        # copied and pasted from the original Conv1D implemnetation
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out) # ... * self.nf

        # if self.multi_lora_b.requires_grad:
        # non_q means k and v
        q, non_q = x[:, :, :self.hidden_size], x[:, :, self.hidden_size:]
        non_q = non_q * self.multi_lora_b.flatten()
        x = torch.cat([q, non_q], dim=2)

        return x


class Conv1DMLPWithIA3(nn.Module):
    def __init__(self, conv1d_layer):
        super().__init__()

        # nf: number of output features; nx: number of input features
        self.nf = conv1d_layer.nf

        self.weight = conv1d_layer.weight
        self.bias = conv1d_layer.bias

        self.multi_lora_b = nn.Parameter(torch.ones(self.nf, 1))

    def forward(self, x):
        # copied and pasted from the original Conv1D implemnetation
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out) # ... * self.nf

        # if self.multi_lora_b.requires_grad:
        x = x * self.multi_lora_b.flatten()

        return x


def modify_with_ia3(transformer, config):
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(config.att_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config.att_layers, c_name):
                    assert isinstance(
                        layer, Conv1D
                    ), f"This code only supports transformers.modeling_utils.Conv1D"
                    # print("Replacing {}.{} with Conv1DAttWithIA3".format(m_name, c_name))
                    setattr(
                        module,
                        c_name,
                        Conv1DAttWithIA3(layer, transformer.config.hidden_size)
                    )
        if re.fullmatch(config.mlp_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config.mlp_layers, c_name):
                    assert isinstance(
                        layer, Conv1D
                    ), f"This code only supports transformers.modeling_utils.Conv1D"
                    # print("Replacing {}.{} with Conv1DMLPWithIA3".format(m_name, c_name))
                    setattr(
                        module,
                        c_name,
                        Conv1DMLPWithIA3(layer)
                    )
    
    return transformer
