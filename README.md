# Mali — Autoregressive Transformer Coded from Scratch in PyTorch

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20ECR%20%7C%20AppRunner-FF9900.svg?logo=amazon-aws&logoColor=white)](https://aws.amazon.com/)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(9%2F9)-brightgreen.svg)](tests/)

**Mali** is a full-stack, causal autoregressive Transformer language model engineered from first principles in raw PyTorch. Designed without high-level wrapper frameworks (no Hugging Face `Trainer`, no LangChain), Mali implements every architectural component down to bare matrix multiplications, custom causal masks, and an MLOps pipeline integrating **Amazon S3** and **AWS App Runner**.

---

## Architecture Overview

```
[Prompt Text]
     │
     ▼
┌────────────────────────────────────────────────────────┐
│  Custom Tokenizer (Char / UTF-8 Byte BPE)             │
└────────────────────────────────────────────────────────┘
     │ Token IDs: (B, T)
     ▼
┌────────────────────────────────────────────────────────┐
│  Token Embedding (wte) + Positional Embedding (wpe)    │
│  (B, T, n_embd)                                        │
└────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────┐
│  Transformer Block (Repeated x4 Layers)                │
│  ├─ RMSNorm (Root Mean Square Normalization)           │
│  ├─ Multi-Head Causal Self-Attention                   │
│  │    Q, K, V Projections                              │
│  │    Scaled Dot-Product: Softmax((Q @ K^T) / sqrt(d)) │
│  │    Lower-Triangular Causal Mask (No future leakage) │
│  │    Dropout + Output Linear Projection               │
│  ├─ Residual Connection (x = x + Attn(RMSNorm(x)))     │
│  ├─ RMSNorm                                            │
│  ├─ Feed-Forward MLP (GELU Activation, 4x Expansion)   │
│  └─ Residual Connection (x = x + MLP(RMSNorm(x)))      │
└────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────┐
│  Final RMSNorm Layer                                   │
│  Language Model Head (Linear: n_embd -> vocab_size)    │
│  (Weight-tied to Token Embedding Matrix)               │
└────────────────────────────────────────────────────────┘
     │
     ▼
[Logits / Next-Token Probability Distribution]
```

### Mathematical Formulation
The core scaled dot-product attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

where the causal mask $M$ is defined as:

$$M_{i,j} = \begin{cases} 0 & \text{if } i \ge j \\ -\infty & \text{if } i < j \end{cases}$$

---

## Engineering Highlights & Technical Depth

- **Zero Future Information Leakage**: Formally tested invariant ensuring modifications to future tokens at position $t+1$ produce zero numerical variation in attention states at position $t$ (see `tests/test_attention.py`).
- **RMSNorm (LLaMA/Mistral Standard)**: Eliminates mean-centering overhead of traditional LayerNorm, scaling activations by root-mean-square to accelerate backward-pass convergence.
- **Weight Tying**: Shares parameters between token embeddings (`wte.weight`) and output projection (`lm_head.weight`), reducing memory consumption by up to 25%.
- **Cosine Learning Rate Decay with Linear Warmup**: Dynamic schedule stabilizing early optimization steps before decaying to $\eta_{\text{min}}$.
- **Inference Sampling Strategies**: Autoregressive decoding featuring **Temperature scaling**, **Top-K truncation**, and **Top-P (Nucleus) sampling**.
- **Production REST API**: FastAPI server exposing sub-50ms inference with automatic OpenAPI / Swagger specifications (`/docs`).
- **AWS Cloud Integration**:
  - **Amazon S3**: Automated artifact registry versioning checkpoints (`s3_sync.py`).
  - **Amazon ECR & App Runner**: Multi-stage Docker containerization ready for cloud-native deployment.

---

## Model Specifications

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Parameters** | **3.20 Million** | Total learnable weights |
| **Context Window ($T$)** | **128 tokens** | Maximum sequence length (`block_size`) |
| **Embedding Dimension ($C$)** | **256** | Hidden state dimension (`n_embd`) |
| **Attention Heads** | **4** | Multi-head channels (`head_dim = 64`) |
| **Layers** | **4 Blocks** | Stacked transformer depth (`n_layer`) |
| **Hardware Acceleration** | **Apple MPS / CUDA / CPU** | Native GPU tensor acceleration |

---

## Project Structure

```
mali/
├── mali/
│   ├── __init__.py           # Package exports
│   ├── config.py             # Hyperparameter dataclass
│   ├── tokenizer.py          # Custom Char & Byte tokenizers
│   ├── model.py              # RMSNorm, CausalSelfAttention, TransformerBlock, MaliLM
│   ├── dataset.py            # Streaming data loader & corpus downloaders
│   ├── train.py              # Training engine, AdamW, cosine schedule, checkpointing
│   └── generate.py           # Text generation with Temperature & Top-P sampling
├── api/
│   ├── app.py                # FastAPI REST server
│   └── Dockerfile            # Container configuration for AWS App Runner
├── cloud/
│   ├── s3_sync.py            # Boto3 S3 artifact manager
│   └── AWS_DEPLOYMENT_GUIDE.md # AWS S3, ECR, and App Runner documentation
├── tests/
│   ├── test_tokenizer.py     # Roundtrip encode/decode & serialization tests
│   ├── test_attention.py     # Causal masking and shape invariants
│   └── test_model.py         # Forward pass, loss computation & backpropagation
├── checkpoints/              # Local weights (.pt) and vocabularies (.json)
├── requirements.txt          # Python dependencies
└── pytest.ini                # Pytest configuration
```

---

## Quickstart

### 1. Setup Environment
```bash
git clone https://github.com/bradleyjoejoseph/mali.git
cd mali

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Test Suite
```bash
pytest -v tests/
```
All 9 unit tests verify attention causality, gradient flow, and serialization integrity.

### 3. Train the Model
Train on the benchmark Tiny Shakespeare corpus (auto-downloads on first run):
```bash
python -m mali.train --iters 1000 --batch-size 32
```
On Apple Silicon (M-series), training runs with native **MPS** GPU acceleration in ~3 minutes.

### 4. Generate Text
```bash
python -m mali.generate --prompt "O Romeo, Romeo," --temp 0.8 --top-p 0.9 --max-tokens 250
```

### 5. Launch the Local Inference API
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```
Visit `http://localhost:8000/docs` to test interactive inference.

---

## AWS Deployment

See [AWS_DEPLOYMENT_GUIDE.md](cloud/AWS_DEPLOYMENT_GUIDE.md) for step-by-step instructions on syncing weights to **Amazon S3** and deploying the containerized API to **AWS App Runner**.

---

## License
MIT License. Created by Bradley Joseph (MEng Computer Science with AI, University of York).
