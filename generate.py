import argparse
import math

import torch

from data import CharTokenizer
from model import GPTConfig, GPT, BigramLM, generate


def load_model(path, device):
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=True
    )

    config = GPTConfig(**checkpoint["config"])
    tokenizer = CharTokenizer(checkpoint["vocab"])

    if checkpoint["model_type"] == "bigram":
        model = BigramLM(config)
    else:
        model = GPT(config)

    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    return model, tokenizer


def complete(model, tokenizer, prompt, args, device):
    tokens = tokenizer.encode(prompt)

    x = torch.tensor(
        [tokens], dtype=torch.long, device=device
    )

    output = generate(
        model,
        x,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

    continuation = output[0, len(tokens):].tolist()

    return prompt + tokenizer.decode(continuation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text using MiniLLM"
    )

    parser.add_argument(
        "--checkpoint", default="runs/gpt/checkpoint.pt"
    )

    parser.add_argument("--prompt", type=str)
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.9)

    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=4)

    args = parser.parse_args()

    if not math.isfinite(args.temperature):
        parser.error("Temperature must be finite")

    if args.temperature != 0 and args.temperature < 0.01:
        parser.error("Temperature must be 0 or at least 0.01")

    if not 0 < args.top_p <= 1:
        parser.error("Top-p must be between 0 and 1")

    if args.tokens <= 0 or args.top_k < 0 or args.threads <= 0:
        parser.error("Invalid generation settings")

    if args.prompt == "":
        parser.error("Prompt cannot be empty")

    torch.set_num_threads(args.threads)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model, tokenizer = load_model(args.checkpoint, device)

    if args.prompt is not None:
        print(complete(
            model, tokenizer, args.prompt, args, device
        ))

    else:
        print("MiniLLM")
        print("Model loaded. Type /quit to exit.")

        while True:
            try:
                prompt = input("\nPrompt: ")

                if prompt == "/quit":
                    break

                if not prompt:
                    continue

                print(complete(
                    model, tokenizer, prompt, args, device
                ))

            except (KeyboardInterrupt, EOFError):
                print()
                break
