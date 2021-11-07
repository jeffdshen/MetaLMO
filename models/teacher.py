# Copyright (c) Jeffrey Shen

"""RoBERTa Model with two heads"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

import models.transformer as T


class TeacherRoBERTa(nn.Module):
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
    ):
        super().__init__()
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

        self.head_x = T.LMHead(
            dim=dim,
            output_tokens=max_tokens,
            activation=activation,
        )
        self.head_y = T.LMHead(
            dim=dim,
            output_tokens=max_tokens,
            activation=activation,
        )
        self.ignore_idx = ignore_idx
        self.apply(lambda mod: T.init_params_bert(mod, 0.02))

    # (N, S), (N, S), (N, S) -> (N, S, O)
    @amp.autocast()
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
        x, y = self.head_x(x), self.head_y(x)
        x, y = x.transpose(0, 1), y.transpose(0, 1)
        return x, y

    # (N, S, O), (N, S) -> (N, S, O)
    def mask_scores(self, x, y, padding_mask):
        return (
            self.head.mask_scores(x, padding_mask),
            self.head.mask_scores(y, padding_mask),
        )

    # (N, S, O) -> (N, S)
    def get_top(self, x, y):
        return self.head_x.get_top(x), self.head_y.get_top(y)

    # ((N, S, O), (N, S), (N, S), K) -> (K, N, S)
    def sample(self, scores_x, scores_y, x, y, mask_x, mask_y, k, alpha=1.0):
        sample_x = self.head_x.sample(scores_x, x, mask_x, k, alpha=alpha)
        sample_y = self.head_x.sample(scores_y, y, mask_y, k, alpha=alpha)
        return sample_x, sample_y

    # (N, S, O) -> (N, S, O*)
    def get_log_prob(self, x, y):
        return self.head_x.get_log_prob(x), self.head_y.get_log_prob(y)

    # (N, S, O) -> (N, S, O*)
    def get_prob(self, x, y):
        return self.head_x.get_prob(x), self.head_y.get_prob(y)

    # ((N, S, O), (N, S), (N, S)) -> (1, )
    def get_loss(self, scores_x, scores_y, x, y, mask_x, mask_y):
        loss_x = self.head_x.get_loss(scores_x, x, mask_x, self.ignore_idx)
        loss_y = self.head_y.get_loss(scores_y, y, mask_y, self.ignore_idx)
        return 0.5 * loss_x + 0.5 * loss_y
