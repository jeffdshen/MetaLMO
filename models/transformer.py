# Copyright (c) Jeffrey Shen

"""Various building blocks for transformer architectures"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedTokenEmbedding(nn.Module):
    """Learned token embedding

    Args:
        num_words (int): Vocab size
        embed_dim (int): Output size of the embedding
        padding_idx (int): Padding index
    """

    def __init__(self, num_words, embed_dim, padding_idx):
        super().__init__()
        self.embed = nn.Embedding(num_words, embed_dim, padding_idx)

    def forward(self, x):
        emb = self.embed(x)
        return emb


class SoftLearnedTokenEmbedding(nn.Module):
    """Soft learned token embedding, capable of

    Args:
        num_words (int): Vocab size
        embed_dim (int): Output size of the embedding
        padding_idx (int): Padding index
    """

    def __init__(self, num_words, embed_dim, padding_idx):
        super().__init__()
        self.embed = nn.Embedding(num_words, embed_dim, padding_idx)
        if padding_idx is None:
            self.padding_idx = None
        else:
            padding_idx = torch.tensor(padding_idx, dtype=torch.long)
            self.padding_idx = nn.Parameter(padding_idx, requires_grad=False)

    @staticmethod
    def soft_forward(x, weight, padding_idx):
        if padding_idx is None:
            return x @ weight
        detached_weight = weight.detach().index_select(0, padding_idx)
        return x @ weight.index_copy(0, padding_idx, detached_weight)

    def forward(self, x):
        if not x.dtype.is_floating_point:
            return self.embed(x)
        else:
            return self.soft_forward(x, self.embed.weight, self.padding_idx)


class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embedding

    Args:
        max_positions (int): Max number of positions
        embed_dim (int): Output size of the embedding
    """

    def __init__(self, max_positions, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(max_positions, embed_dim)

    def forward(self, x):
        emb = self.embed(x)
        return emb


def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("Unsupported activation function: {}".format(activation))


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, dim, n_heads, ff_dim, activation, dropout, attn_dropout, act_dropout
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=attn_dropout)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_attn_norm = nn.LayerNorm(dim)

        self.linear1 = nn.Linear(dim, ff_dim)
        self.activation = get_activation_fn(activation)
        self.act_dropout = nn.Dropout(act_dropout)
        self.linear2 = nn.Linear(ff_dim, dim)
        self.ff_dropout = nn.Dropout(dropout)

        self.ff_norm = nn.LayerNorm(dim)

    # ((S, N, E), (N, S), (S, S)) -> (S, N, E)
    def forward(self, x, key_padding_mask=None, attn_mask=None):
        residual = x
        x, _ = self.self_attn(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        x = self.self_attn_dropout(x)
        x = residual + x
        x = self.self_attn_norm(x)

        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.act_dropout(x)
        x = self.linear2(x)
        x = self.ff_dropout(x)
        x = residual + x
        x = self.ff_norm(x)

        return x


class TransformerPrenormEncoderLayer(nn.Module):
    def __init__(
        self, dim, n_heads, ff_dim, activation, dropout, attn_dropout, act_dropout
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=attn_dropout)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_attn_norm = nn.LayerNorm(dim)

        self.linear1 = nn.Linear(dim, ff_dim)
        self.activation = get_activation_fn(activation)
        self.act_dropout = nn.Dropout(act_dropout)
        self.linear2 = nn.Linear(ff_dim, dim)
        self.ff_dropout = nn.Dropout(dropout)

        self.ff_norm = nn.LayerNorm(dim)

    # ((S, N, E), (N, S), (S, S)) -> (S, N, E)
    def forward(self, x, key_padding_mask=None, attn_mask=None):
        residual = x
        x = self.self_attn_norm(x)
        x, _ = self.self_attn(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        x = self.self_attn_dropout(x)
        x = residual + x

        residual = x
        x = self.ff_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.act_dropout(x)
        x = self.linear2(x)
        x = self.ff_dropout(x)
        x = residual + x

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, layer, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for i in range(n_layers)])

    # ((S, N, E), (N, S), (S, S)) -> (S, N, E)
    def forward(self, x, key_padding_mask=None, attn_mask=None):
        output = x
        for layer in self.layers:
            output = layer(
                output, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        return output


class TransformerEncoderEmbedding(nn.Module):
    def __init__(self, embed_tokens, embed_position, dim, dropout, layer_norm=True):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.embed_position = embed_position
        if layer_norm:
            self.layer_norm = nn.LayerNorm(dim)
        else:
            self.layer_norm = None
        self.dropout = nn.Dropout(dropout)

    # ((S, N), (S, N)) -> (S, N, E)
    def forward(self, x, positions):
        x = self.embed_tokens(x)
        x = x + self.embed_position(positions)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class LMHead(nn.Module):
    def __init__(self, dim, output_tokens, activation, weight=None):
        super().__init__()
        self.ff_linear = nn.Linear(dim, dim)
        self.activation = get_activation_fn(activation)
        self.layer_norm = nn.LayerNorm(dim)

        self.output = nn.Linear(dim, output_tokens)
        if weight is not None:
            self.output.weight = weight

    # (S, N, E) -> (S, N, O)
    def forward(self, x):
        x = self.ff_linear(x)
        x = self.activation(x)
        x = self.layer_norm(x)

        x = self.output(x)
        return x

    # (N, S, O), (N, S) -> (N, S, O)
    @staticmethod
    def mask_scores(x, padding_mask):
        # Swap the dimension back to original order for speed reasons
        swap = not x.is_contiguous()
        if swap:
            x = x.transpose(0, 1)
            padding_mask = padding_mask.transpose(0, 1)
        masked = x.masked_fill(padding_mask.unsqueeze(-1), float("-inf"))
        if swap:
            return masked.transpose(0, 1)
        return masked

    # (N, S, O) -> (N, S)
    @staticmethod
    def get_top(x):
        return torch.argmax(x, dim=-1)

    # ((N, S, O), (N, S), (N, S), K) -> (K, N, S)
    @staticmethod
    def sample(scores, x, mask, k, alpha=1.0):
        x = x.detach().clone()
        scores = scores.detach().clone()
        x = x.unsqueeze(-1).repeat(1, 1, k)
        x[~mask, :] = softmax_sample(scores[~mask, :], k, alpha=alpha)
        x = x.permute(-1, 0, 1).contiguous()
        return x

    # (N, S, O) -> (N, S, O*)
    @staticmethod
    def get_log_prob(x):
        return F.log_softmax(x, dim=-1)

    # (N, S, O) -> (N, S, O*)
    @staticmethod
    def get_prob(x):
        return F.softmax(x, dim=-1)

    # ((N, S, O), (N, S), (N, S)) -> (1, )
    @staticmethod
    def get_loss(scores, y, mask, ignore_idx):
        y = y.masked_fill(mask, ignore_idx)
        # Swap the dimension back to original order for speed reasons
        if not scores.is_contiguous():
            scores = scores.transpose(0, 1)
            y = y.transpose(0, 1)
        return F.cross_entropy(scores.transpose(1, -1), y, ignore_index=ignore_idx)


# (S, N) -> (S, N=1)
def get_positions(x):
    positions = torch.arange(x.shape[0], dtype=torch.long, device=x.device)
    return positions.unsqueeze(1)


# (N, S) -> (N, S)
def get_padding_mask(x, padding_idx):
    return x.eq(padding_idx)


# (N, O) -> (N, K)
def softmax_sample(x, k, alpha=1.0):
    s = F.softmax(x / alpha, dim=-1)
    m = torch.multinomial(s.view(-1, s.size(-1)), k, replacement=True)
    return m.view(s.size(0), k)


def init_params_bert(module, std):
    """
    BERT initialization using normal distribution.
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        if module.weight.requires_grad:
            nn.init.normal_(module.weight, std=std)

    if isinstance(module, nn.Embedding):
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].fill_(0)

    if isinstance(module, nn.Linear):
        if module.bias is not None:
            nn.init.zeros_(module.bias)
