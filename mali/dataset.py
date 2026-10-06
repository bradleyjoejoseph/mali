import os
import requests
import torch
from typing import Tuple
from .config import MaliConfig
from .tokenizer import CharTokenizer

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def load_or_download_data(data_path: str = "data/input.txt") -> str:
    """Download Tiny Shakespeare corpus if not already cached locally."""
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    if not os.path.exists(data_path):
        print(f"[Dataset] Downloading benchmark corpus from {SHAKESPEARE_URL}...")
        response = requests.get(SHAKESPEARE_URL, timeout=15)
        response.raise_for_status()
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"[Dataset] Saved {len(response.text):,} characters to {data_path}")
    
    with open(data_path, "r", encoding="utf-8") as f:
        return f.read()


class DataLoader:
    """
    Batched data loader for autoregressive language modeling.
    Extracts random chunks of length block_size with targets shifted by +1 token.
    """
    def __init__(self, data: str, tokenizer: CharTokenizer, train_split: float = 0.9):
        self.tokenizer = tokenizer
        encoded_tokens = torch.tensor(tokenizer.encode(data), dtype=torch.long)
        
        n = int(train_split * len(encoded_tokens))
        self.train_data = encoded_tokens[:n]
        self.val_data = encoded_tokens[n:]
        print(f"[DataLoader] Split corpus: {len(self.train_data):,} train tokens, {len(self.val_data):,} val tokens.")

    def get_batch(self, split: str, config: MaliConfig, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch a batch of input sequences (x) and target sequences (y)."""
        data = self.train_data if split == "train" else self.val_data
        # Generate random starting indices for the batch
        ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
        x = torch.stack([data[i:i + config.block_size] for i in ix])
        # Target y is shifted by 1 token forward
        y = torch.stack([data[i + 1:i + config.block_size + 1] for i in ix])
        return x.to(device), y.to(device)
