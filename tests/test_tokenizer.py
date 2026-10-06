import os
import tempfile
import pytest
from mali.tokenizer import CharTokenizer, ByteTokenizer

def test_char_tokenizer_roundtrip():
    text = "Hello, Mali! Testing custom tokenization."
    chars = list(set(text))
    tok = CharTokenizer(chars)
    
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    
    assert decoded == text
    assert len(encoded) == len(text)
    assert tok.vocab_size == len(chars)

def test_char_tokenizer_serialization():
    text = "Machine Learning Engineering"
    tok = CharTokenizer(list(set(text)))
    
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_file = os.path.join(tmpdir, "vocab.json")
        tok.save(vocab_file)
        
        loaded_tok = CharTokenizer.load(vocab_file)
        assert loaded_tok.vocab_size == tok.vocab_size
        assert loaded_tok.encode(text) == tok.encode(text)

def test_byte_tokenizer():
    text = "Transformers 🚀 in PyTorch 🔥"
    tok = ByteTokenizer()
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    
    assert decoded == text
    assert tok.vocab_size == 256
