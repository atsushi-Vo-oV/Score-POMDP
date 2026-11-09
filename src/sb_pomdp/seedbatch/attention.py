"""Self attention expressed with vmap-compatible tensor operations.

Torch 2.3 has no batching rule for either CPU flash SDPA or math SDPA. Its
fallback runs attention once per seed. This equivalent dropout-free attention
retains the original parameter names, while batching both matrix products.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class BatchedSelfAttention(nn.Module):
    def __init__(self, original):
        super().__init__()
        if not original.batch_first or not original._qkv_same_embed_dim:
            raise ValueError("seedbatch attention needs batch-first self attention")
        self.embed_dim = original.embed_dim
        self.num_heads = original.num_heads
        self.batch_first = True
        self._qkv_same_embed_dim = True
        self.in_proj_weight = original.in_proj_weight
        self.in_proj_bias = original.in_proj_bias
        self.out_proj = original.out_proj

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        if key_padding_mask is not None or attn_mask is not None or is_causal or need_weights:
            raise ValueError("seedbatch particle attention does not support masks or attention weights")
        if query is not key or query is not value:
            raise ValueError("seedbatch particle attention requires self attention")
        batch, length, width = query.shape
        heads = self.num_heads
        depth = width // heads
        projected = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        q, k, v = [part.reshape(batch, length, heads, depth).transpose(1, 2)
                   for part in projected.chunk(3, dim=-1)]
        weights = torch.softmax((q / math.sqrt(depth)) @ k.transpose(-2, -1), dim=-1)
        output = (weights @ v).transpose(1, 2).reshape(batch, length, width)
        return F.linear(output, self.out_proj.weight, self.out_proj.bias), None


def use_batched_attention(module):
    """Replace attention only on the private functional template."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            setattr(module, name, BatchedSelfAttention(child))
        else:
            use_batched_attention(child)
