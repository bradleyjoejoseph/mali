import os
import argparse
import torch
from .config import MaliConfig
from .tokenizer import CharTokenizer
from .model import MaliLM

def load_mali_model(checkpoint_path: str = "checkpoints/mali_best.pt", vocab_path: str = "checkpoints/mali_vocab.json", device: str = None):
    """Load model weights and tokenizer from checkpoint files."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first via: python -m mali.train")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary not found at {vocab_path}.")

    tokenizer = CharTokenizer.load(vocab_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config: MaliConfig = checkpoint["config"]
    config.vocab_size = tokenizer.vocab_size

    model = MaliLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    selected_device = device if device else config.get_device()
    model.to(selected_device)
    model.eval()
    return model, tokenizer, selected_device


def generate_text(
    model: MaliLM,
    tokenizer: CharTokenizer,
    prompt: str = "\n",
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    device: str = "cpu"
) -> str:
    """Generate text from a prompt string."""
    encoded = tokenizer.encode(prompt)
    if not encoded:
        encoded = [0]
    
    idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    generated_indices = model.generate(
        idx,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )[0].tolist()
    
    return tokenizer.decode(generated_indices)


def main():
    parser = argparse.ArgumentParser(description="Generate text using trained Mali Transformer")
    parser.add_argument("--prompt", type=str, default="O Romeo, Romeo,", help="Input conditioning prompt")
    parser.add_argument("--max-tokens", type=int, default=250, help="Max tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature (lower = deterministic, higher = creative)")
    parser.add_argument("--top-k", type=int, default=40, help="Top-K token truncation")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-P nucleus sampling threshold")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/mali_best.pt", help="Path to checkpoint")
    parser.add_argument("--vocab", type=str, default="checkpoints/mali_vocab.json", help="Path to vocabulary")
    args = parser.parse_args()

    model, tokenizer, device = load_mali_model(args.checkpoint, args.vocab)
    print(f"\nPrompt: \"{args.prompt}\"")
    print(f"--- [Mali Generation: Temp={args.temp}, Top-K={args.top_k}, Top-P={args.top_p}] ---")
    output = generate_text(
        model, tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device
    )
    print(output)
    print("------------------------------------------------------------------------\n")

if __name__ == "__main__":
    main()
