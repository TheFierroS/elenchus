"""The function encoder: token ids in, one unit vector per function out.

A small Transformer trained from scratch. Two heads share one body:

  embed()        pools the body's output into a vector and projects it onto
                 the unit sphere; similar functions should land close
  mlm_logits()   predicts every position's token, used only in pre-training

Size is deliberately modest. Train holds about 16,650 distinct functions;
a model with capacity far beyond that memorises them instead of learning
what they have in common. The default (4 layers, width 256) is about 3.5M
parameters, and larger configurations are an experiment to measure, not an
assumption.

Padding must never change a function's vector. Attention is masked with the
padding mask, and mean pooling averages only real positions, so a function
embedded alone and the same function padded inside a batch of longer ones
give the same vector - a property tested directly.
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class EncoderConfig:
    vocab_size: int
    pad_id: int = 0
    max_len: int = 1024
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    embed_dim: int = 128
    pooling: str = "mean"  # "mean" or "cls"; docs/experiments.md E7

    def to_dict(self):
        return asdict(self)


class FunctionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.pooling not in ("mean", "cls"):
            raise ValueError(f"unknown pooling {config.pooling!r}")
        self.config = config

        self.tokens = nn.Embedding(config.vocab_size, config.d_model,
                                   padding_idx=config.pad_id)
        self.positions = nn.Embedding(config.max_len, config.d_model)
        layer = nn.TransformerEncoderLayer(
            config.d_model, config.n_heads, config.d_ff, config.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.body = nn.TransformerEncoder(layer, config.n_layers,
                                          enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.d_model)
        self.projection = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.embed_dim),
        )
        # Tied to the input embedding: predicting a token and reading it use
        # the same notion of what the token is, and it saves a vocab-sized
        # matrix.
        self.mlm_bias = nn.Parameter(torch.zeros(config.vocab_size))

    def hidden(self, ids, mask):
        """Contextual vectors, [batch, length, d_model]."""
        length = ids.shape[1]
        if length > self.config.max_len:
            raise ValueError(f"sequence of {length} exceeds max_len "
                             f"{self.config.max_len}")
        positions = torch.arange(length, device=ids.device)
        x = self.tokens(ids) + self.positions(positions)[None, :, :]
        x = self.body(x, src_key_padding_mask=~mask)
        return self.norm(x)

    def embed(self, ids, mask):
        """One unit-length vector per sequence, [batch, embed_dim]."""
        h = self.hidden(ids, mask)
        if self.config.pooling == "cls":
            pooled = h[:, 0]
        else:
            weights = mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return F.normalize(self.projection(pooled), dim=-1)

    def mlm_logits(self, ids, mask):
        """Token predictions at every position, [batch, length, vocab]."""
        h = self.hidden(ids, mask)
        return h @ self.tokens.weight.T + self.mlm_bias


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
