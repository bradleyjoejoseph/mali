import torch
import pytest
from mali.config import MaliConfig
from mali.model import CausalSelfAttention, RMSNorm

def test_rmsnorm():
    norm = RMSNorm(dim=64)
    x = torch.randn(2, 10, 64)
    out = norm(x)
    assert out.shape == (2, 10, 64)
    assert not torch.isnan(out).any()

def test_causal_self_attention_shapes():
    config = MaliConfig(n_embd=64, n_head=4, block_size=32)
    attn = CausalSelfAttention(config)
    x = torch.randn(2, 16, 64) # Batch=2, Seq=16, Embd=64
    out = attn(x)
    
    assert out.shape == (2, 16, 64)
    assert not torch.isnan(out).any()

def test_causal_mask_prevents_future_leakage():
    """Verify that earlier outputs are unaffected by changes to future inputs."""
    config = MaliConfig(n_embd=64, n_head=4, block_size=32, dropout=0.0)
    attn = CausalSelfAttention(config)
    attn.eval() # Disable dropout for deterministic output

    x1 = torch.randn(1, 10, 64)
    x2 = x1.clone()
    # Change the very last token in sequence 2
    x2[:, -1, :] = torch.randn(1, 1, 64)

    with torch.no_grad():
        out1 = attn(x1)
        out2 = attn(x2)

    # All positions except the modified last token must be numerically identical
    torch.testing.assert_close(out1[:, :-1, :], out2[:, :-1, :], atol=1e-5, rtol=1e-5)
