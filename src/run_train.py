"""Execute the improved Tiny Shakespeare transformer training pipeline."""
import math
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Torch {torch.__version__} | threads={torch.get_num_threads()}")

# --- Config ---
block_size = 64
n_embd = 128
n_head = 4
n_layer = 6
dropout = 0.2

batch_size = 32
learning_rate = 1e-3
max_iters = 5000
warmup_iters = 200
eval_interval = 250
eval_iters = 40
grad_clip = 1.0

temperature = 0.8
top_k = 40

checkpoint_path = "model.pt"
BASELINE_TEST_LOSS = 1.9125
BASELINE_PARAMS_K = 209.7

# --- Data ---
with open("tiny shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

vocab = sorted(list(set(text)))
vocab_size = len(vocab)
encod = {ch: i for i, ch in enumerate(vocab)}
decod = {i: ch for i, ch in enumerate(vocab)}
encode = lambda s: [encod[i] for i in s]
decode = lambda l: "".join([decod[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_set = data[:n]
test_set = data[n:]
print(f"vocab={vocab_size} train={len(train_set)} test={len(test_set)}")


def get_batch(split):
    source = train_set if split == "train" else test_set
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i : i + block_size] for i in ix])
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# --- Model ---
class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


def top_k_logits(logits, k):
    if k is None or k <= 0:
        return logits
    v, _ = torch.topk(logits, min(k, logits.size(-1)))
    return logits.masked_fill(logits < v[:, [-1]], float("-inf"))


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.blocks(tok_emb + pos_emb)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            logits = top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx


model = LanguageModel(vocab_size, n_embd, n_head, n_layer, block_size, dropout).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Params: {n_params / 1e3:.1f}K (baseline was {BASELINE_PARAMS_K}K)")

start_context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("--- BEFORE TRAINING ---")
print(decode(model.generate(start_context.clone(), max_new_tokens=200, temperature=1.0, top_k=None)[0].tolist()))


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "test"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    progress = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    return learning_rate * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, betas=(0.9, 0.99), weight_decay=0.1
)

train_losses = []
test_losses = []
best_test = float("inf")
t0 = time.time()

for i in range(max_iters):
    lr = get_lr(i)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    if i % eval_interval == 0 or i == max_iters - 1:
        losses = estimate_loss()
        train_losses.append(losses["train"])
        test_losses.append(losses["test"])
        elapsed = time.time() - t0
        print(
            f"Iteration {i}: Train {losses['train']:.4f} | Test {losses['test']:.4f} | "
            f"LR {lr:.2e} | {elapsed / 60:.1f}m"
        )
        if losses["test"] < best_test:
            best_test = losses["test"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {
                        "vocab_size": vocab_size,
                        "n_embd": n_embd,
                        "n_head": n_head,
                        "n_layer": n_layer,
                        "block_size": block_size,
                        "dropout": dropout,
                    },
                    "test_loss": best_test,
                },
                checkpoint_path,
            )

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

print(f"Final step loss: {loss.item():.4f}")
print(f"Best test loss: {best_test:.4f}  (baseline was {BASELINE_TEST_LOSS:.4f})")
print(f"Improvement: {BASELINE_TEST_LOSS - best_test:.4f} lower test loss")

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="train_loss")
plt.plot(test_losses, label="test_loss")
plt.axhline(BASELINE_TEST_LOSS, color="gray", linestyle="--", label=f"baseline test ({BASELINE_TEST_LOSS})")
plt.xlabel(f"Evaluation step (every {eval_interval} iterations)")
plt.ylabel("Loss")
plt.title("Training Progress")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=120)
print("Saved loss_curve.png")

ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
cfg = ckpt["config"]
loaded_model = LanguageModel(
    cfg["vocab_size"],
    cfg["n_embd"],
    cfg["n_head"],
    cfg["n_layer"],
    cfg["block_size"],
    cfg["dropout"],
).to(device)
loaded_model.load_state_dict(ckpt["model"])
loaded_model.eval()

print("--- AFTER TRAINING ---")
torch.manual_seed(42)
print(
    decode(
        loaded_model.generate(
            start_context.clone(),
            max_new_tokens=1000,
            temperature=temperature,
            top_k=top_k,
        )[0].tolist()
    )
)

print("\n=== COMPARISON ===")
print(f"Baseline test loss : {BASELINE_TEST_LOSS:.4f}")
print(f"Improved test loss : {ckpt['test_loss']:.4f}")
print(f"Delta              : {BASELINE_TEST_LOSS - ckpt['test_loss']:+.4f}")
print(f"Baseline params    : {BASELINE_PARAMS_K}K")
print(f"Improved params    : {n_params / 1e3:.1f}K")
print(f"Context length     : 32 -> {block_size}")
print(f"Embedding / layers : 64x4 -> {n_embd}x{n_layer}")

with open("results_comparison.txt", "w", encoding="utf-8") as f:
    f.write(f"baseline_test_loss={BASELINE_TEST_LOSS}\n")
    f.write(f"improved_test_loss={ckpt['test_loss']}\n")
    f.write(f"delta={BASELINE_TEST_LOSS - ckpt['test_loss']}\n")
    f.write(f"params_k={n_params / 1e3}\n")
print("Saved results_comparison.txt")
