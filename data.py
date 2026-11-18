from pathlib import Path
from urllib.request import urlretrieve

import torch

DATA_PATH = Path("data/tinyshakespeare.txt")

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/"
    "data/tinyshakespeare/input.txt"
)

def download_dataset():
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(DATA_URL, DATA_PATH)
        print("Dataset downloaded")

    return DATA_PATH


class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))

        self.stoi = {
            ch: i + 1 for i, ch in enumerate(self.chars)
        }

        self.itos = {
            i: ch for ch, i in self.stoi.items()
        }

        self.vocab_size = len(self.chars) + 1

    def encode(self, text):
        return [self.stoi.get(ch, 0) for ch in text]

    def decode(self, tokens):
        return "".join(self.itos.get(i, "�") for i in tokens)


def load_data(path=DATA_PATH, tokenizer=None):
    path = Path(path)

    if not path.exists() and path == DATA_PATH:
        download_dataset()

    text = path.read_text(encoding="utf-8")

    split = int(0.9 * len(text))
    train_text = text[:split]
    val_text = text[split:]

    if tokenizer is None:
        tokenizer = CharTokenizer(train_text)

    train_data = torch.tensor(
        tokenizer.encode(train_text), dtype=torch.long
    )

    val_data = torch.tensor(
        tokenizer.encode(val_text), dtype=torch.long
    )

    return train_data, val_data, tokenizer


def get_batch(data, batch_size, block_size, device):
    if len(data) <= block_size:
        raise ValueError("Dataset is smaller than the context window")

    indices = torch.randint(
        len(data) - block_size, (batch_size,)
    ).tolist()

    x = torch.stack([
        data[i:i + block_size] for i in indices
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1] for i in indices
    ])

    return x.to(device), y.to(device)
