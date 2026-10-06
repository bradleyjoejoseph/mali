import os
import time
import math
import argparse
import torch
from .config import MaliConfig
from .tokenizer import CharTokenizer
from .dataset import load_or_download_data, DataLoader
from .model import MaliLM

def get_lr(it: int, config: MaliConfig) -> float:
    """Cosine learning rate decay with linear warmup."""
    # 1) Linear warmup for warmup_iters steps
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    # 2) If it > max_iters, return min learning rate
    if it > config.max_iters:
        return config.min_lr
    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (it - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def estimate_loss(model: MaliLM, loader: DataLoader, config: MaliConfig, device: str) -> dict:
    """Evaluate model on multiple batches of train and val splits to get a clean loss estimate."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            x, y = loader.get_batch(split, config, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(args):
    # 1. Setup Configuration & Device
    config = MaliConfig()
    device = args.device if args.device else config.get_device()
    config.max_iters = args.iters
    config.batch_size = args.batch_size
    print(f"\n==========================================")
    print(f"🚀 Initializing Mali Training Engine")
    print(f"Hardware Device Target: {device.upper()}")
    print(f"Target Iterations: {config.max_iters} | Batch Size: {config.batch_size}")
    print(f"==========================================\n")

    # 2. Load Data & Build Tokenizer
    raw_data = load_or_download_data(args.data)
    tokenizer = CharTokenizer(list(raw_data))
    config.vocab_size = tokenizer.vocab_size
    print(f"[Tokenizer] Built vocabulary of {tokenizer.vocab_size} unique tokens.")

    # Save tokenizer vocabulary immediately
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    vocab_path = os.path.join(config.checkpoint_dir, "mali_vocab.json")
    tokenizer.save(vocab_path)
    print(f"[Tokenizer] Vocabulary serialized to {vocab_path}")

    # Prepare Data Loader
    loader = DataLoader(raw_data, tokenizer)

    # 3. Instantiate Transformer Model
    model = MaliLM(config)
    model.to(device)
    num_params = model.get_num_params()
    print(f"[Model] MaliLM Architecture Initialized with {num_params:,} parameters ({num_params/1e6:.2f}M).")

    # 4. Configure AdamW Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95)
    )

    # 5. Training Loop
    best_val_loss = float("inf")
    t0 = time.time()

    for it in range(config.max_iters + 1):
        # Update learning rate according to cosine schedule
        lr = get_lr(it, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Evaluate periodically
        if it % config.eval_interval == 0 or it == config.max_iters:
            losses = estimate_loss(model, loader, config, device)
            dt = time.time() - t0
            t0 = time.time()
            print(f"Step {it:4d}/{config.max_iters} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | LR: {lr:.2e} | Time: {dt:.2f}s")

            # Checkpoint on best validation loss
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                checkpoint_path = os.path.join(config.checkpoint_dir, "mali_best.pt")
                torch.save({
                    "step": it,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "config": config,
                }, checkpoint_path)
                print(f"  ⭐ Saved new best model checkpoint to {checkpoint_path} (Val Loss: {best_val_loss:.4f})")

                # Optional AWS S3 sync hook
                if args.sync_s3:
                    try:
                        from cloud.s3_sync import upload_to_s3
                        upload_to_s3(checkpoint_path, config.s3_bucket, f"checkpoints/mali_step_{it}.pt")
                    except Exception as e:
                        print(f"  [AWS Warning] S3 sync skipped: {e}")

            # Sample text preview
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            sampled_tokens = model.generate(context, max_new_tokens=60, temperature=0.8)[0].tolist()
            preview_text = tokenizer.decode(sampled_tokens).replace("\n", " ")
            print(f"  Preview: \"{preview_text[:70]}...\"\n")

        # Fetch batch and forward
        x, y = loader.get_batch("train", config, device)
        logits, loss = model(x, y)

        # Backward pass & optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping prevents exploding gradients
        optimizer.step()

    print(f"\n✅ Training complete! Best validation loss achieved: {best_val_loss:.4f}")
    return model, tokenizer, best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Train Mali Transformer Language Model")
    parser.add_argument("--iters", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda, mps)")
    parser.add_argument("--data", type=str, default="data/input.txt", help="Path to training corpus")
    parser.add_argument("--sync-s3", action="store_true", help="Sync best checkpoints to AWS S3")
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
