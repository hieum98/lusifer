import torch
import torch.nn as nn
import einops
from transformers.models.bert.modeling_bert import BertAttention
from transformers.models.bert.configuration_bert import BertConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa


class FFWithAddedTokens(nn.Module):
    def __init__(
            self, 
            in_dim: int,
            out_dim: int, 
            num_added_tokens: int=1, 
            model_dtype=torch.bfloat16
            ):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(in_dim, in_dim, dtype=model_dtype),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim, dtype=model_dtype),
        )
        self.num_added_tokens = num_added_tokens
        if num_added_tokens > 0:
            self.added_tokens = nn.Parameter(torch.randn(num_added_tokens, out_dim, dtype=model_dtype))
    
    def forward(self, x, **kwargs):
        x = self.ff(x)
        if self.num_added_tokens > 0:
            added_tokens = einops.repeat(self.added_tokens, 'n d -> b n d', b=x.size(0))
            x = torch.cat([x, added_tokens], dim=1) # (b, n + n_a, d)
        return x
    

class AttentionPooling(nn.Module):
    def __init__(
            self, 
            in_dim: int, 
            out_dim: int, 
            num_query_tokens: int=1,
            model_dtype=torch.bfloat16
            ):
        super().__init__()
        attn_config = BertConfig(
            hidden_size=in_dim,
            num_attention_heads=8,
            attention_probs_dropout_prob=0.1,
            _attn_implementation='sdpa',
            is_decoder=False
        )
        self.attn = BertAttention(config=attn_config)
        self.query_tokens = nn.Parameter(torch.randn(num_query_tokens, in_dim, dtype=model_dtype))
        self.ff = nn.Sequential(
            nn.Linear(in_dim, in_dim, dtype=model_dtype),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim, dtype=model_dtype),
        )
    
    def forward(
            self,
            x, # (b, n, d)
            attention_mask, # (b, n)
            **kwargs
            ):
        query_tokens = einops.repeat(self.query_tokens, 'n d -> b n d', b=x.size(0)) # (b, n_q, d)
        encoder_extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
            mask=attention_mask,
            dtype=x.dtype,
            tgt_len=query_tokens.size(1),
        ) # (b, 1, n_q, n)
        attention_output = self.attn(
            hidden_states=query_tokens,
            attention_mask=None,
            encoder_hidden_states=x,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        attention_output = attention_output[0] # (b, n_q, d)
        x = self.ff(attention_output)
        return x # (b, n_q, d)
        


