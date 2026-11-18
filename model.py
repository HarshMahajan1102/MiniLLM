import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 3
    dropout: float = 0.1

    def __post_init__(self):
        if min(
            self.vocab_size, self.block_size,
            self.n_embd, self.n_head, self.n_layer
        ) < 1:
            raise ValueError("Model dimensions must be positive")

        if self.n_embd % self.n_head != 0:
            raise ValueError(
                "Embedding size must be divisible by number of heads"
            )

        if not 0 <= self.dropout < 1:
            raise ValueError("Invalid dropout probability")


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head

        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        self.projection = nn.Linear(
            config.n_embd, config.n_embd
        )

        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    config.block_size,
                    config.block_size,
                    dtype=torch.bool
                )
            ),
            persistent=False
        )

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Split embedding dimension into attention heads
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_size)

        # Prevent attention to future tokens
        scores = scores.masked_fill(
            ~self.mask[:T, :T], float("-inf")
        )

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = weights @ v

        # Combine attention heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.projection(out)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attention = MultiHeadAttention(config)

        self.ffn = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd)
        )

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))

        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.n_embd
        )

        self.position_embedding = nn.Embedding(
            config.block_size, config.n_embd
        )

        self.blocks = nn.Sequential(*[
            TransformerBlock(config)
            for _ in range(config.n_layer)
        ])

        self.ln_final = nn.LayerNorm(config.n_embd)

        self.lm_head = nn.Linear(
            config.n_embd, config.vocab_size, bias=False
        )

        self.dropout = nn.Dropout(config.dropout)

        self.apply(self.init_weights)

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, x):
        B, T = x.shape

        positions = torch.arange(T, device=x.device)

        token_emb = self.token_embedding(x)
        position_emb = self.position_embedding(positions)

        x = self.dropout(token_emb + position_emb)
        x = self.blocks(x)
        x = self.ln_final(x)

        return self.lm_head(x)


class BigramLM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.embedding = nn.Embedding(
            config.vocab_size, config.vocab_size
        )

    def forward(self, x):
        return self.embedding(x)


def sample_next(logits, temperature, top_k, top_p):
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k > 0:
        k = min(top_k, logits.size(-1))

        values, _ = torch.topk(logits, k)

        logits = logits.masked_fill(
            logits < values[..., -1:], float("-inf")
        )

    if top_p < 1:
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1
        )

        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)

        remove = cumulative >= top_p

        # Retain the token that crosses the threshold
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False

        sorted_logits = sorted_logits.masked_fill(
            remove, float("-inf")
        )

        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(-1, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)

    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(model, x, max_new_tokens,
             temperature=0.8, top_k=0, top_p=1.0):

    model.eval()

    for _ in range(max_new_tokens):
        context = x[:, -model.config.block_size:]

        logits = model(context)
        logits = logits[:, -1, :]

        # Do not generate the unknown character
        logits[:, 0] = float("-inf")

        next_token = sample_next(
            logits, temperature, top_k, top_p
        )

        x = torch.cat((x, next_token), dim=1)

    return x
