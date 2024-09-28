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
        self.dtype = model_dtype
        self.ff = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
        )
        self.num_added_tokens = num_added_tokens
        if num_added_tokens > 0:
            self.added_tokens = nn.Parameter(torch.randn(num_added_tokens, out_dim))
    
    def forward(self, x, **kwargs):
        # Cast to correct device type
        with torch.autocast(device_type=x.device.type, dtype=self.dtype):
            x = self.ff(x)
            if self.num_added_tokens > 0:
                added_tokens = einops.repeat(self.added_tokens, 'n d -> b n d', b=x.size(0))
                x = torch.cat([x, added_tokens], dim=1) # (b, n + n_a, d)
            x = x.to(dtype=self.dtype)
        return x

