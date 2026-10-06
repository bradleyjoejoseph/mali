import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from .config import MaliConfig

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (used in LLaMA, Mistral, Gemma).
    Simpler and faster than LayerNorm by skipping mean-centering.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2) + eps)
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention mechanism from scratch.
    Implements: Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(d_k) + causal_mask ) @ V
    """
    def __init__(self, config: MaliConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dim must be divisible by n_head"
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Key, Query, Value projections combined into a single linear layer for efficiency
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection back to residual stream
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask buffer: lower triangular matrix ensures token i cannot attend to token j > i
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size() # Batch size, Sequence length, Embedding dim

        # Calculate Query, Key, Value vectors for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape to (Batch, n_head, Sequence_length, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Apply causal mask: mask out future tokens with -inf
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted combination of values
        y = att @ v # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Re-assemble head outputs

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    Position-wise Feed-Forward Network.
    Projects embeddings 4x wide with GELU non-linearity, then back to original dimension.
    """
    def __init__(self, config: MaliConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Block.
    x = x + Attention(Norm(x))
    x = x + MLP(Norm(x))
    """
    def __init__(self, config: MaliConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MaliLM(nn.Module):
    """
    Full GPT-style Autoregressive Causal Language Model.
    """
    def __init__(self, config: MaliConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))

        # Language Model Head projects hidden states back to vocabulary logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight Tying: tie token embeddings and output projection weights
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize all weights with standard Gaussian
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, max block size is {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device) # (t)

        # Token + Positional embeddings
        tok_emb = self.transformer.wte(idx) # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Forward through all transformer blocks
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # If targets are provided, calculate Cross-Entropy loss
            logits = self.lm_head(x) # (b, t, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            # Inference optimization: only forward the last token logit
            logits = self.lm_head(x[:, [-1], :]) # (b, 1, vocab_size)
            return logits, None

    def get_num_params(self) -> int:
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95
    ) -> torch.Tensor:
        """
        Autoregressively generate new tokens given a conditioning context.
        Supports Temperature scaling, Top-K filtering, and Top-P (Nucleus) sampling.
        """
        for _ in range(max_new_tokens):
            # Crop context if it exceeds maximum block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature # Scale by temperature

            # Top-K filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Top-P (Nucleus) sampling
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above top_p threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep at least one token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float("Inf")

            # Softmax to get probabilities and sample next token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled token to running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
