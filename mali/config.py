from dataclasses import dataclass
import torch

@dataclass
class MaliConfig:
    """Hyperparameter configuration for Mali Transformer."""
    
    # Model Architecture
    vocab_size: int = 256      # Dynamically updated by tokenizer
    block_size: int = 128      # Context window length (maximum sequence length)
    n_embd: int = 256          # Embedding dimension
    n_head: int = 4            # Number of attention heads (head_dim = n_embd // n_head)
    n_layer: int = 4           # Number of Transformer blocks
    dropout: float = 0.1       # Dropout probability for regularization
    bias: bool = False         # Modern standard: bias=False in Linear layers improves throughput
    
    # Training Parameters
    batch_size: int = 32       # Batch size for SGD
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 100
    max_iters: int = 2000
    eval_interval: int = 200
    eval_iters: int = 40
    weight_decay: float = 0.01
    
    # Checkpointing & AWS
    checkpoint_dir: str = "checkpoints"
    s3_bucket: str = "mali-model-artifacts"
    
    @classmethod
    def get_device(cls) -> str:
        """Automatically select Apple Silicon MPS, CUDA GPU, or CPU."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"
