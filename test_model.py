import torch
import torch.nn.functional as F

from data import CharTokenizer, get_batch
from model import GPTConfig, GPT, BigramLM, generate


def test_tokenizer():
    text = "hello world"
    tokenizer = CharTokenizer(text)

    tokens = tokenizer.encode(text)

    assert tokenizer.decode(tokens) == text
    assert tokenizer.encode("Z") == [0]

    print("Tokenizer test passed")


def test_batch():
    data = torch.arange(100)

    x, y = get_batch(data, 4, 8, "cpu")

    assert x.shape == (4, 8)
    assert y.shape == (4, 8)

    assert torch.equal(x[:, 1:], y[:, :-1])

    print("Batch generation test passed")


def test_forward():
    config = GPTConfig(
        vocab_size=30,
        block_size=16,
        n_embd=32,
        n_head=4,
        n_layer=2
    )

    model = GPT(config)

    x = torch.randint(0, 30, (4, 16))

    logits = model(x)

    assert logits.shape == (4, 16, 30)

    print("Forward pass test passed")


def test_causal_attention():
    config = GPTConfig(
        vocab_size=30,
        block_size=16,
        n_embd=32,
        n_head=4,
        n_layer=2
    )

    model = GPT(config)
    model.eval()

    x1 = torch.randint(0, 30, (1, 16))
    x2 = x1.clone()

    # Change only future tokens
    x2[:, 8:] = (x2[:, 8:] + 1) % config.vocab_size

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    assert torch.allclose(
        out1[:, :8],
        out2[:, :8],
        atol=1e-6
    )

    print("Causal attention test passed")


def test_training():
    config = GPTConfig(
        vocab_size=10,
        block_size=8,
        n_embd=32,
        n_head=4,
        n_layer=2,
        dropout=0.0
    )

    model = GPT(config)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.003
    )

    x = torch.randint(1, 10, (4, 8))
    y = (x + 1) % 10

    def get_loss():
        logits = model(x)

        return F.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            y.reshape(-1)
        )

    initial_loss = get_loss().item()

    for _ in range(50):
        loss = get_loss()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = get_loss().item()

    assert final_loss < initial_loss

    print(
        f"Training test passed: "
        f"{initial_loss:.4f} -> {final_loss:.4f}"
    )


def test_generation():
    config = GPTConfig(
        vocab_size=30,
        block_size=16,
        n_embd=32,
        n_head=4,
        n_layer=2
    )

    x = torch.tensor([[1, 2, 3]])

    for model in [GPT(config), BigramLM(config)]:
        output = generate(
            model, x, max_new_tokens=20,
            temperature=0.8, top_k=5, top_p=0.9
        )

        assert output.shape == (1, 23)
        assert torch.equal(output[:, :3], x)
        assert (output[:, 3:] > 0).all()
        assert (output < config.vocab_size).all()

    print("Generation test passed")


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.set_num_threads(2)

    test_tokenizer()
    test_batch()
    test_forward()
    test_causal_attention()
    test_training()
    test_generation()

    print("\nAll tests passed")
