import argparse

import torch

from src.config import config
from src.data import load_text, build_vocab, get_tokenizer
from src.model import LanguageModel


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = LanguageModel(
        checkpoint["vocab_size"],
        checkpoint["n_embd"],
        checkpoint["n_head"],
        checkpoint["n_layer"],
        checkpoint["block_size"],
        checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def generate(checkpoint_path=None, max_new_tokens=500):
    checkpoint_path = checkpoint_path or config.checkpoint_path
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = load_text()
    vocab, _ = build_vocab(text)
    encode, decode = get_tokenizer(vocab)

    model = load_model(checkpoint_path, device)
    start_context = torch.zeros((1, 1), dtype=torch.long, device=device)
    output = model.generate(start_context, max_new_tokens=max_new_tokens)[0].tolist()
    return decode(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=config.checkpoint_path)
    parser.add_argument("--tokens", type=int, default=500)
    args = parser.parse_args()

    print(generate(args.checkpoint, args.tokens))
