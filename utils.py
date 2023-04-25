from pathlib import Path

import numpy as np
import torch

VOCAB_SIZE=187
CHARS =  ['n', 'T', '£', 'W', 'x', 'ù', 'f', 'É', '到', 'æ', '•', '²', '┌', '┤', 'Z', 'S', '…', '/', 'R', 'E', '+', 'o', '∂', 'ü', 'g', '≠', '╫', '午', '‘', '（', 'e', '“', '<', 'ç', '达', '"', '’', 'ₙ', '#', 'P', '½', 'A', 't', 'X', 'β', 'θ', 'ø', '└', '≡', '║', 'Q', '，', '═', 'h', 'λ', '{', 'ρ', '0', '\t', 'ö', '：', 'K', '∣', '╩', 'b', 'π', '┬', '.', ')', 'Δ', '?', '8', '7', ']', 'é', 'q', '-', 'v', '┘', ' ', 'c', '∑', 'u', 'p', 'N', '≈', 'C', 'ł', '■', 'Ü', '*', '|', 'U', 'è', '！', 'd', 'm', '│', '–', '┼', "'", 'ã', '}', 'y', '_', '→', '\n', 'z', 'G', 's', '=', 'ů', '2', '!', '\u2009', '^', 'Y', '⬇', '精', '╥', '4', '9', ':', '下', '≤', '>', '\\', 'a', '√', 'V', 'D', ';', 'B', 'î', '~', '￼', '₹', 'L', '—', '(', '5', '↑', 'ñ', 'ò', '├', 'O', 'ń', 'i', 'γ', '┐', '[', '$', 'w', '`', 'µ', '1', 'l', '%', '≥', '─', '✅', '）', 'í', '”', 'j', '−', '€', 'r', 'º', 'ℏ', 'F', '6', '3', ',', 'J', '┴', '&', '×', '@', 'I', 'à', '度', 'φ', '维', 'H', 'k', 'M']
INDEX_TO_CHAR = {i:c for i, c in enumerate(CHARS)}
CHAR_TO_INDEX = {c:i for i, c in enumerate(CHARS)}
def encode(seq):
    return [CHAR_TO_INDEX[c] for c in seq ]

def decode(indexes):
    return "".join([INDEX_TO_CHAR[int(i)] for i in indexes])

def save_train_test_set():
    with open(get_root() /"data/python_text.txt", "r") as f:
        data = f.read()
    train = data[:int(len(data)*0.9)]
    test = data[int(len(data)*0.9):]

    torch.save(torch.tensor(data=encode(train), dtype=torch.long), get_root() / "data/train.pt")
    torch.save(torch.tensor(data=encode(test), dtype=torch.long), get_root() / "data/test.pt")

def get_root():
    return Path(__file__).parent

def get_batch(data, batch_size, block_size):
    idx = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+1+block_size] for i in idx])
    return x, y
