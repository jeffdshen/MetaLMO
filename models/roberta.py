# Copyright (c) Jeffrey Shen

"""RoBERTa Model with LM head"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

import models.transformer as T


class RoBERTa(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        ff_dim,
        activation,
        dropout,
        attn_dropout,
        act_dropout,
        n_layers,
        max_positions,
        max_tokens,
        padding_idx,
        ignore_idx,
        prenorm=False,
        soft_embedding=False,
    ):
        super().__init__()
        if soft_embedding:
            embed_tokens = T.SoftLearnedTokenEmbedding(max_tokens, dim, padding_idx)
        else:
            embed_tokens = T.LearnedTokenEmbedding(max_tokens, dim, padding_idx)
        self.embed = T.TransformerEncoderEmbedding(
            embed_tokens,
            T.LearnedPositionalEmbedding(max_positions, dim),
            dim=dim,
            dropout=dropout,
            layer_norm=(not prenorm),
        )

        if not prenorm:
            encoder_layer = T.TransformerEncoderLayer(
                dim=dim,
                n_heads=n_heads,
                ff_dim=ff_dim,
                activation=activation,
                dropout=dropout,
                attn_dropout=attn_dropout,
                act_dropout=act_dropout,
            )
        else:
            encoder_layer = T.TransformerPrenormEncoderLayer(
                dim=dim,
                n_heads=n_heads,
                ff_dim=ff_dim,
                activation=activation,
                dropout=dropout,
                attn_dropout=attn_dropout,
                act_dropout=act_dropout,
            )

        self.encoder = T.TransformerEncoder(
            encoder_layer,
            n_layers=n_layers,
        )

        if prenorm:
            self.final_layer_norm = nn.LayerNorm(dim)
        else:
            self.final_layer_norm = None

        self.head = T.LMHead(
            dim=dim,
            output_tokens=max_tokens,
            activation=activation,
            weight=embed_tokens.embed.weight,
        )
        self.ignore_idx = ignore_idx
        self.apply(lambda mod: T.init_params_bert(mod, 0.02))

    # (N, S), (N, S), (N, S) -> (N, S, O)
    def forward(self, x, positions=None, padding_mask=None):
        x = x.transpose(0, 1)
        if positions is None:
            positions = T.get_positions(x)
        else:
            positions = positions.transpose(0, 1)
        x = self.embed(x, positions)
        x = self.encoder.forward(x, key_padding_mask=padding_mask)
        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)
        x = self.head(x)
        x = x.transpose(0, 1)
        return x

    # (N, S, O), (N, S) -> (N, S, O)
    def mask_scores(self, x, padding_mask):
        return self.head.mask_scores(x, padding_mask)

    # (N, S, O) -> (N, S)
    def get_top(self, x):
        return self.head.get_top(x)

    # ((N, S, O), (N, S), (N, S), K) -> (K, N, S)
    def sample(self, scores, x, mask, k, alpha=1.0):
        return self.head.sample(scores, x, mask, k, alpha=alpha)

    # (N, S, O) -> (N, S, O*)
    def get_log_prob(self, x):
        return self.head.get_log_prob(x)

    # (N, S, O) -> (N, S, O*)
    def get_prob(self, x):
        return self.head.get_prob(x)

    # ((N, S, O), (N, S), (N, S)) -> (1, )
    def get_loss(self, scores, y, mask):
        return self.head.get_loss(scores, y, mask, self.ignore_idx)
