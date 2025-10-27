import torch
import torch.nn.functional as F
import torch.nn as nn
import weakref

# class MetanetworkTransformer(nn.Module):
#     def __init__(self, output_dim, cfg):
#         super().__init__()
#         transformer_cfg = cfg.metanetwork.transformer_cfg
#         self.fc_in = nn.Linear(cfg.hidden_size * cfg.num_mem_token, transformer_cfg.encoder_cfg.d_model)
#         encoder_layer = nn.TransformerEncoderLayer(**transformer_cfg.encoder_cfg)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_cfg.num_layers)
#         self.fc_out1 = nn.Linear(transformer_cfg.encoder_cfg.d_model * cfg.num_layers, transformer_cfg.output_bottleneck)
#         self.fc_out2 = nn.Linear(transformer_cfg.output_bottleneck, output_dim)
        
#         self.num_layers = cfg.num_layers
#         self.num_mem_token = cfg.num_mem_token
#         self.hidden_size = cfg.hidden_size

#     def forward(self, memory_states:torch.Tensor) -> dict:
#         '''
#         memory_states: (batch_size, num_layer, num_mem_token, hidden_size)
#         '''
#         batch_size = memory_states.shape[0]
#         x = memory_states.view(batch_size, self.num_layers, self.num_mem_token * self.hidden_size)  
#         x = F.gelu(self.fc_in(x))  # (batch_size, num_layer, d_model)    
#         x = self.transformer_encoder(x)
#         x = x.contiguous().view(batch_size, -1)  # (batch_size, num_layer * d_model) 
#         x = F.gelu(self.fc_out1(x))
#         x = self.fc_out2(x)  # (batch_size, output_dim)
#         return x

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

class Metanetwork(nn.Module):
    def __init__(self, metamodel:nn.Module, cfg, output_dim: int):
        super().__init__()
        self.lora_r = cfg.model.lora_r
        self.output_dim = output_dim
        self.metamodel = metamodel
        if cfg.metanetwork.type == "transformer":
            self.metanetwork = MetanetworkTransformer(cfg)
            self.scale = cfg.metanetwork.transformer_cfg.scale
        else:
            raise ValueError(f"Unknown metanetwork type: {cfg.metanetwork.type}")

    @torch.compile
    def forward(self, input_ids, input_attention_mask, evidence_ids, evidence_attention_mask, metalora = None, labels = None, use_metanet = True, **kwargs) -> dict:
        '''
        memory_states: (batch_size, num_layer, num_mem_token, hidden_size)
        '''
        if use_metanet:
            assert metalora is not None, "metalora cannot be None when use_metanet is True"
            loradict = self.generate_lora_dict(evidence_ids, evidence_attention_mask, metalora)
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
        
    
