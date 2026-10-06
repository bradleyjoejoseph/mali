import torch
import pytest
from mali.config import MaliConfig
from mali.model import MaliLM

def test_model_forward_and_loss():
    config = MaliConfig(vocab_size=50, block_size=32, n_embd=64, n_head=4, n_layer=2)
    model = MaliLM(config)

    # 1. Forward without targets (inference mode)
    idx = torch.randint(0, 50, (2, 10))
    logits, loss = model(idx)
    assert logits.shape == (2, 1, 50)
    assert loss is None

    # 2. Forward with targets (training mode)
    targets = torch.randint(0, 50, (2, 10))
    logits, loss = model(idx, targets)
    assert logits.shape == (2, 10, 50)
    assert loss is not None
    assert loss.item() > 0.0

def test_model_gradients_flow():
    config = MaliConfig(vocab_size=50, block_size=32, n_embd=64, n_head=4, n_layer=2)
    model = MaliLM(config)
    idx = torch.randint(0, 50, (2, 10))
    targets = torch.randint(0, 50, (2, 10))

    _, loss = model(idx, targets)
    loss.backward()

    # Verify gradients exist and are not NaN
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient is None for {name}"
            assert not torch.isnan(param.grad).any(), f"Gradient has NaN in {name}"

def test_model_generation():
    config = MaliConfig(vocab_size=50, block_size=32, n_embd=64, n_head=4, n_layer=2)
    model = MaliLM(config)
    model.eval()

    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out = model.generate(prompt, max_new_tokens=15, temperature=1.0)
    assert out.shape == (1, 18) # 3 prompt tokens + 15 generated tokens
