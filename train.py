"""Tiny character-level language model, trained from random weights."""

import torch
from torch import nn


class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {char: i for i, char in enumerate(self.chars)}

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return ''.join(self.chars[i] for i in ids)


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, block_size=32, width=32, heads=2):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(block_size, width)
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        self.final_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab_size)

    def forward(self, tokens):
        batch, length = tokens.shape
        if length > self.block_size:
            raise ValueError('Input exceeds block_size')
        positions = torch.arange(length, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        mask = torch.triu(torch.ones(length, length, device=tokens.device, dtype=torch.bool), diagonal=1)
        normalized = self.norm1(hidden)
        attended, _ = self.attention(normalized, normalized, normalized, attn_mask=mask, need_weights=False)
        hidden = hidden + attended
        hidden = hidden + self.feed_forward(self.norm2(hidden))
        return self.output(self.final_norm(hidden))


def train_steps(model, data, steps=250, block_size=32, batch_size=16, learning_rate=0.003):
    """Train next-character prediction; return before/after loss on a fixed batch."""
    if data.numel() <= block_size + 1:
        raise ValueError('Corpus must be longer than block_size + 1')
    if block_size != model.block_size:
        raise ValueError('block_size must match model')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def batch():
        starts = torch.randint(data.numel() - block_size, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in starts])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
        return x, y

    x_eval, y_eval = batch()
    def eval_loss():
        model.eval()
        with torch.no_grad():
            logits = model(x_eval)
            result = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y_eval.reshape(-1)).item()
        return result

    initial = eval_loss()
    model.train()
    for _ in range(steps):
        x, y = batch()
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return [initial, eval_loss()]


def generate(model, tokenizer, prompt, max_new_tokens=80, temperature=0.8):
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(tokens[:, -model.block_size:])[:, -1] / temperature
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)
    return tokenizer.decode(tokens[0].tolist())
