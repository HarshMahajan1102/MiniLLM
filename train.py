import argparse
import csv
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from data import DATA_PATH, CharTokenizer, load_data, get_batch
from model import GPTConfig, GPT, BigramLM


@torch.no_grad()
def evaluate(model, data, batch_size, device, batches=20):
    model.eval()

    losses = []

    for _ in range(batches):
        x, y = get_batch(
            data, batch_size, model.config.block_size, device
        )

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1)
        )

        losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)


@torch.no_grad()
def full_validation(model, data, device, batch_size=32):
    model.eval()

    block_size = model.config.block_size
    total_loss = 0
    total_tokens = 0

    # Evaluate every held-out target once
    for start in range(0, len(data) - 1, batch_size * block_size):
        end = min(
            start + batch_size * block_size,
            len(data) - 1
        )

        count = end - start
        full = count // block_size

        if full > 0:
            stop = start + full * block_size

            x = data[start:stop].reshape(full, block_size)
            y = data[start + 1:stop + 1].reshape(full, block_size)

            x, y = x.to(device), y.to(device)

            logits = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="sum"
            )

            total_loss += loss.item()
            total_tokens += y.numel()

        tail_start = start + full * block_size

        if tail_start < end:
            x = data[tail_start:end].unsqueeze(0).to(device)
            y = data[tail_start + 1:end + 1].unsqueeze(0).to(device)

            logits = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="sum"
            )

            total_loss += loss.item()
            total_tokens += y.numel()

    loss = total_loss / total_tokens

    return loss, math.exp(loss)


def save_checkpoint(path, model, optimizer, tokenizer, step, model_type):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(model.config),
        "vocab": tokenizer.chars,
        "model_type": model_type,
        "step": step
    }, path)


def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output / "checkpoint.pt"
    metrics_path = output / "metrics.csv"

    start_step = 0
    tokenizer = None
    checkpoint = None
    model_type = args.model

    if args.resume:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )

        tokenizer = CharTokenizer(checkpoint["vocab"])
        model_type = checkpoint["model_type"]
        start_step = checkpoint["step"]

    elif checkpoint_path.exists():
        raise ValueError(
            "Checkpoint already exists. Use --resume or another --out"
        )

    train_data, val_data, tokenizer = load_data(
        args.data, tokenizer
    )

    if checkpoint is not None:
        config = GPTConfig(**checkpoint["config"])
    else:
        config = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=args.block_size,
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_layer=args.n_layer,
            dropout=args.dropout
        )

    if min(len(train_data), len(val_data)) <= config.block_size:
        raise ValueError("Dataset is too small for the chosen block size")

    if args.steps <= start_step:
        raise ValueError("--steps must exceed the saved step")

    if model_type == "bigram":
        model = BigramLM(config)
    else:
        model = GPT(config)

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr
    )

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        for group in optimizer.param_groups:
            group["lr"] = args.lr

    params = sum(p.numel() for p in model.parameters())

    print(f"Device: {device}")
    print(f"Model: {model_type}")
    print(f"Parameters: {params:,}")
    print(f"Vocabulary: {tokenizer.vocab_size}")
    print(f"Training characters: {len(train_data):,}")
    print(f"Validation characters: {len(val_data):,}")

    start_time = time.time()

    mode = "a" if args.resume and metrics_path.exists() else "w"

    with open(metrics_path, mode, newline="") as file:
        writer = csv.writer(file)

        if mode == "w":
            writer.writerow([
                "step", "train_loss", "val_loss", "perplexity"
            ])

        if not args.resume:
            initial_loss, initial_ppl = full_validation(
                model, val_data, device
            )

            print(
                f"Initial validation loss: {initial_loss:.4f} | "
                f"Perplexity: {initial_ppl:.2f}"
            )

        model.train()

        running_loss = 0
        updates = 0

        for step in range(start_step + 1, args.steps + 1):
            x, y = get_batch(
                train_data,
                args.batch_size,
                config.block_size,
                device
            )

            logits = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                y.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()

            running_loss += loss.item()
            updates += 1

            if step % args.eval_every == 0 or step == args.steps:
                train_loss = running_loss / updates

                val_loss = evaluate(
                    model, val_data, args.batch_size, device
                )

                perplexity = math.exp(val_loss)

                print(
                    f"Step {step:5d} | "
                    f"Train loss: {train_loss:.4f} | "
                    f"Val loss: {val_loss:.4f} | "
                    f"PPL: {perplexity:.2f}"
                )

                writer.writerow([
                    step, train_loss, val_loss, perplexity
                ])

                file.flush()

                save_checkpoint(
                    checkpoint_path, model, optimizer,
                    tokenizer, step, model_type
                )

                running_loss = 0
                updates = 0

    val_loss, perplexity = full_validation(
        model, val_data, device
    )

    print(f"\nFinal validation loss: {val_loss:.4f}")
    print(f"Final perplexity: {perplexity:.2f}")
    print(f"Elapsed time: {time.time() - start_time:.2f}s")
    print(f"Checkpoint saved: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MiniLLM")

    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--model", choices=["gpt", "bigram"], default="gpt")
    parser.add_argument("--out", default="runs/gpt")

    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=64)

    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--eval-every", type=int, default=100)

    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    if min(args.steps, args.batch_size, args.eval_every, args.threads) <= 0:
        parser.error("Training settings must be positive")

    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("Learning rate must be positive and finite")

    train(args)
