import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Dict


class MetanetworkTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_layers = cfg.num_layers
        self.num_mem_token = cfg.num_mem_token
        self.hidden_size = cfg.hidden_size

        self.layer_pe = nn.Parameter(torch.zeros((self.num_layers, self.hidden_size)), requires_grad=True)
        self.token_pe = nn.Parameter(torch.zeros((self.num_mem_token, self.hidden_size)), requires_grad=True)

        transformer_cfg = cfg.metanetwork.transformer_cfg
        self.transformer_layers = nn.ModuleList([nn.TransformerEncoderLayer(**transformer_cfg.encoder_cfg) for _ in range(transformer_cfg.num_layers)])


    def forward(self, memory_states:torch.Tensor) -> dict:
        '''
        memory_states: (batch_size, num_layer, num_mem_token, hidden_size)
        '''
        memory_states = memory_states + self.layer_pe.unsqueeze(-2) + self.token_pe # apply PE
        batch_size = memory_states.shape[0]
        for i in range(len(self.transformer_layers)):
            if i % 2 == 0:
                memory_states = self.transformer_layers[i](memory_states.transpose(1, 2).flatten(0, 1)).unflatten(0, (batch_size, self.num_mem_token)).transpose(1, 2) # exchange information among layers
            else:
                memory_states = self.transformer_layers[i](memory_states.flatten(0, 1)).unflatten(0, (batch_size, self.num_layers)) # exchange information among tokens
        return memory_states.flatten(1, -1)


def build_module_from_tree(tree: dict | torch.Tensor):
    if torch.is_tensor(next(iter(tree.values()))):
        return nn.ParameterDict(tree)
    if isinstance(next(iter(tree.keys())), int):
        return nn.ModuleList([build_module_from_tree(tree[i]) for i in range(len(tree))])
    else:
        return nn.ModuleDict({k: build_module_from_tree(v) for k, v in tree.items()})


class Metanetwork(nn.Module):
    def __init__(self, metamodel:nn.Module, cfg, output_dim: int):
        super().__init__()
        self.lora_r = cfg.model.lora_r
        self.output_dim = output_dim
        self.metamodel = metamodel
        metalora_dict = metamodel.init_lora_dict(cfg.model.lora_r, scale=cfg.metanetwork.transformer_cfg.scale, device=metamodel.device)
        self.metalora = build_module_from_tree(metalora_dict)
        if cfg.metanetwork.type == "transformer":
            self.metanetwork = MetanetworkTransformer(cfg)
            self.scale = cfg.metanetwork.transformer_cfg.scale
        else:
            raise ValueError(f"Unknown metanetwork type: {cfg.metanetwork.type}")
        
    @property
    def config(self):
        # Prefer live inner config if present; else fall back to cached copy
        return getattr(self.metamodel, "config", None)

    # @torch.compile # (mode="max-autotune")
    def forward(self, input_ids, input_attention_mask, evidence_ids, evidence_attention_mask, labels = None, use_metanet = True, **kwargs) -> dict:
        '''
        memory_states: (batch_size, num_layer, num_mem_token, hidden_size)
        '''
        if use_metanet:
            loradict = self.generate_lora_dict(evidence_ids, evidence_attention_mask, self.metalora)
            outputs = self.metamodel(input_ids=input_ids, attention_mask=input_attention_mask, loradict=loradict, labels=labels, ignore_mem_token=True, **kwargs)
        else:
            outputs = self.metamodel(input_ids=input_ids, attention_mask=input_attention_mask, labels=labels, ignore_mem_token=True, **kwargs)
        return outputs
    
    def generate_lora_dict(self, evidence_ids, evidence_attention_mask, metalora):
        outputs = self.metamodel(input_ids=evidence_ids, attention_mask=evidence_attention_mask, loradict=metalora)
        memory_states = outputs.memory_states
        plain_output = self.metanetwork(memory_states)  # (batch_size, output_dim)
        loradict = self.metamodel.generate_lora_dict(self.lora_r, scale=self.scale, plain_tensor=plain_output)
        return loradict
        
    
