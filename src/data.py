import torch

from src.config import config


def load_text(path=None):
    path = path or config.data_path
    with open(path, "r") as f:
        return f.read()


def build_vocab(text):
    vocab = sorted(list(set(text)))
    return vocab, len(vocab)


def get_tokenizer(vocab):
    encod = {ch: i for i, ch in enumerate(vocab)}
    decod = {i: ch for i, ch in enumerate(vocab)}
    encode = lambda s: [encod[i] for i in s]
    decode = lambda l: ''.join([decod[i] for i in l])
    return encode, decode


def train_test_split(data, split=0.9):
    n = int(split * len(data))
    return data[:n], data[n:]


def get_batch(source, block_size, batch_size, device):
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i: i + block_size] for i in ix])
    y = torch.stack([source[i + 1: i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def prepare_data():
    text = load_text()
    vocab, vocab_size = build_vocab(text)
    encode, decode = get_tokenizer(vocab)
    data = torch.tensor(encode(text), dtype=torch.long)
    train_set, test_set = train_test_split(data)
    return {
        "vocab_size": vocab_size,
        "encode": encode,
        "decode": decode,
        "train_set": train_set,
        "test_set": test_set,
    }
