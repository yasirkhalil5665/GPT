from dataclasses import dataclass


@dataclass
class Config:
    block_size: int = 32
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1

    batch_size: int = 32
    learning_rate: float = 3e-4
    max_iters: int = 10000
    eval_interval: int = 500
    eval_iters: int = 100
    grad_clip: float = 1.0

    seed: int = 42
    data_path: str = "data/tiny_shakespeare.txt"
    checkpoint_path: str = "checkpoints/model.pt"


config = Config()
