import os

import torch

from src.config import config
from src.data import prepare_data, get_batch
from src.model import LanguageModel


@torch.no_grad()
def estimate_loss(model, train_set, test_set, device):
    out = {}
    model.eval()
    for name, source in [("train", train_set), ("test", test_set)]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            xb, yb = get_batch(source, config.block_size, config.batch_size, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def train():
    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data = prepare_data()
    vocab_size = data["vocab_size"]
    train_set = data["train_set"]
    test_set = data["test_set"]

    model = LanguageModel(
        vocab_size, config.n_embd, config.n_head, config.n_layer, config.block_size, config.dropout
    ).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e3:.1f}K")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_iters)

    train_losses, test_losses = [], []

    for i in range(config.max_iters):
        if i % config.eval_interval == 0 or i == config.max_iters - 1:
            losses = estimate_loss(model, train_set, test_set, device)
            train_losses.append(losses["train"])
            test_losses.append(losses["test"])
            print(f"Iteration {i}: Train Loss {losses['train']:.4f} | Test Loss {losses['test']:.4f}")

        xb, yb = get_batch(train_set, config.block_size, config.batch_size, device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

    print(f"Final Loss: {loss.item():.4f}")

    os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "n_embd": config.n_embd,
            "n_head": config.n_head,
            "n_layer": config.n_layer,
            "block_size": config.block_size,
            "dropout": config.dropout,
        },
        config.checkpoint_path,
    )
    print(f"Saved checkpoint to {config.checkpoint_path}")

    return train_losses, test_losses


if __name__ == "__main__":
    train()
