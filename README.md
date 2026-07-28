# Mini Transformer 

A character-level language model trained on the Tiny Shakespeare dataset, built up from a simple bigram baseline to a small Transformer (multi-head self-attention, feed-forward blocks, residual connections, layer norm, dropout).

## Project Structure

```
bigram-transformer/
├── data/
│   └── tiny_shakespeare.txt
├── src/
│   ├── __init__.py
│   ├── config.py       # hyperparameters (block_size, n_embd, n_head, n_layer, lr, etc.)
│   ├── data.py          # loading text, vocab, encode/decode, train/test split, batching
│   ├── model.py         # Head, MultiHeadAttention, FeedForward, Block, BigramLanguageModel, LanguageModel
│   ├── train.py          # training loop, evaluation, checkpoint saving
│   └── generate.py       # loading a checkpoint and sampling text
├── notebooks/
│   |── baseline.ipynb  # My code actually(Sums up the whole thing)
│   |── better_model.ipynb  # My code with better results
|   └── exploration.ipynb  # experimentation/visualization only, imports from src/
├── checkpoints/
|   ├── loss_curve.png         # saved model weights (gitignored)
|   |── result_comparsion.txt   #Baseline and updated moodel
|   └── train_log.txt       
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.9+
- See [`requirements.txt`](requirements.txt): `torch`, `matplotlib`, `jupyter`

```bash
pip install -r requirements.txt
```

## Dataset

Download the Tiny Shakespeare dataset into `data/`:

```bash
curl -o data/tiny_shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Usage

Run everything from the project root.

**Train:**

```bash
python -m src.train
```

Saves a checkpoint to `checkpoints/model.pt`.

**Generate from a checkpoint:**

```bash
python -m src.generate --checkpoint checkpoints/model.pt --tokens 500
```

**Explore interactively:**

```bash
jupyter notebook notebooks/exploration.ipynb
```

## Configuration

All hyperparameters live in `src/config.py` (`Config` dataclass) — edit values there rather than in `train.py` or `model.py`.

## Notes

- `train.py` averages evaluation loss over `eval_iters` batches for a stable estimate, uses a cosine LR schedule, and clips gradients.
- Checkpoints store the model weights plus the architecture hyperparameters needed to reconstruct the model in `generate.py`.
- `notebooks/exploration.ipynb` contains no model/data logic — it only imports and calls functions from `src/`.
