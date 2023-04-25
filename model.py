

import torch.nn as nn
import torch






class Head(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        head_size = config.embedding_dim//config.n_head
        self.key = nn.Linear(config.embedding_dim, head_size, bias=False)
        self.query = nn.Linear(config.embedding_dim, head_size, bias=False)
        self.value = nn.Linear(config.embedding_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        att = (q @ k.transpose(-2, -1)) *  C**-0.5
        att = att.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        att = torch.functional.F.softmax(att, dim=-1)
        att =self.dropout(att)
        #att = self.attn_dropout(att)
        y = att @ v 
        return y
    
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return torch.functional.F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in  range(config.n_head)])
        self.proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(x))
    
class FeedForward(nn.Module):
    def __init__(self, embedding_dim) -> None:
        super().__init__()
        self.l1 = nn.Linear(embedding_dim, 4*embedding_dim, bias=True)
        self.l2 = nn.Linear(4*embedding_dim, embedding_dim, bias=True)
        self.dropout=nn.Dropout(0.1)
    def forward(self, x):
        x = self.l2(torch.functional.F.relu(self.l1(x)))
        return self.dropout(x)
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffw = FeedForward(embedding_dim=config.embedding_dim)
        self.ln_1 = LayerNorm(config.embedding_dim)
        self.ln_2 = LayerNorm(config.embedding_dim)
    def forward(self, x):
        x = x + self.sa(self.ln_1(x))
        x = x + self.ffw(self.ln_2(x))
        return x

class Transformer(nn.Module):
    """Only considers last token to make predictions"""
    def __init__(self, config) -> None:
        super().__init__()
        self.config=config
        self.token_embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_dim)
        self.positional_encoding = nn.Embedding(num_embeddings=config.block_size, embedding_dim=config.embedding_dim)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_blocks)],
            LayerNorm(ndim=config.embedding_dim),
        )
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
    def forward(self, x):
        x = self.token_embedding(x) + self.positional_encoding(torch.arange(self.config.block_size, device=x.device))
        x = self.blocks(x)
        x = self.lm_head(x)
        # Cross entropy already have softmax
        return x

    def generate_sequence(self, idx, max_tokens, block_size):
        with torch.no_grad():
            for i in range(max_tokens):
                # crop last block_size tokens
                idx_cond = idx[:,-block_size:]
                # softmax here is in the other dimension, because we are starting with an actual batch prediction (1,T,vocab)
                probs = self(idx_cond).softmax(dim=-1)
                # selecting only last token
                idx_next = torch.multinomial(input=probs[:,-1,:], num_samples=1)
                idx = torch.cat((idx, idx_next), dim=-1)
            return idx.squeeze()
                












