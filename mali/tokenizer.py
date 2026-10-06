import json
from typing import List, Dict

class CharTokenizer:
    """
    Character-level tokenizer built from scratch.
    Maps characters to integer IDs and back, with serialization for production serving.
    """
    def __init__(self, chars: List[str] = None):
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        if chars is not None:
            self.build_vocab(chars)

    def build_vocab(self, chars: List[str]) -> None:
        """Create mapping from a sorted list of unique characters."""
        sorted_chars = sorted(list(set(chars)))
        self.stoi = {ch: i for i, ch in enumerate(sorted_chars)}
        self.itos = {i: ch for i, ch in enumerate(sorted_chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> List[int]:
        """Convert a string into a list of integer token IDs."""
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, tokens: List[int]) -> str:
        """Convert a list of integer token IDs back into text."""
        return "".join([self.itos.get(i, "") for i in tokens])

    def save(self, filepath: str) -> None:
        """Serialize the vocabulary mapping to a JSON file."""
        data = {
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
            "vocab_size": self.vocab_size
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "CharTokenizer":
        """Load a saved vocabulary from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.stoi = data["stoi"]
        tok.itos = {int(k): v for k, v in data["itos"].items()}
        return tok


class ByteTokenizer:
    """
    Byte-level UTF-8 tokenizer.
    Guarantees no Out-of-Vocabulary (OOV) errors by mapping raw UTF-8 bytes directly to 0..255.
    """
    @property
    def vocab_size(self) -> int:
        return 256

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: List[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")
