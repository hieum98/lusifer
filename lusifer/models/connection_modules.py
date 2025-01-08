from typing import Optional
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
            print(f'Adding {num_added_tokens} trainable tokens to FFWithAddedTokens')
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


class ProbalisticTokenEmbedding(nn.Embedding):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:
            return super().forward(tokens)
        return torch.matmul(tokens, self.weight)

    def reset_parameters(self, mean=0., std=1.) -> None:
        torch.nn.init.normal_(self.weight, mean=mean, std=std)
        self._fill_padding_idx_with_zero()


class EmbeddingTable(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int, 
            vocab_size: Optional[int],
            padding_idx: Optional[int]=None,
            llm_embedding: Optional[nn.Embedding]=None,
            model_dtype=torch.bfloat16
            ) -> None:
        super().__init__()
        self.dtype = model_dtype
        self.ff = nn.Sequential(
            nn.Linear(in_dim, vocab_size),
            nn.Softmax(dim=-1),
        )
        print(f'Initializing EmbeddingTable with vocab_size={vocab_size}, out_dim={out_dim}, padding_idx={padding_idx}')
        self.embedding = ProbalisticTokenEmbedding(vocab_size, out_dim, padding_idx=padding_idx)
        if llm_embedding is not None:
            # Initialize from LLM embedding
            assert llm_embedding.weight.size() == self.embedding.weight.size(), "embedding sizes must match!"
            self.embedding.weight.data = llm_embedding.weight.data

    def forward(self, x, **kwargs):
        # Cast to correct device type
        with torch.autocast(device_type=x.device.type, dtype=self.dtype):
            x = self.ff(x) # (b, n, vocab_size)
            x = self.embedding(x)
            x = x.to(dtype=self.dtype) # (b, n, d)
        return x

