from pathlib import Path

import torch
import torch.nn.functional as F

with open(Path(__file__).parent / "names.txt", "r") as f:
    words = f.read().splitlines()

# build sting -> int and int -> string
chars = sorted(set("".join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i + 1: s for i, s in enumerate(chars)}
itos[0] = "."

alphabet_size = len(stoi)


# build dataset
block_size = 3  # context-length: how many chars on each try are parsed in


def build_dataset(words):
    X, Y = [], []

    for w in words:
        context = [0] * block_size
        for ch in w:
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


# MLP
embd_dim = 10  # dimension of embedding vektor
hidden_dim = 200  # hidden layer dimension

C = torch.randn((alphabet_size, embd_dim))
W1 = torch.randn((embd_dim * block_size, hidden_dim))
b1 = torch.randn(hidden_dim)
W2 = torch.randn((hidden_dim, alphabet_size))
b2 = torch.randn(alphabet_size)

params = [C, W1, W2, b1, b2]
for p in params:
    p.requires_grad = True

# work
max_steps = 2000
batch_size = 32
learning_rate = 0.1

def main():
    X, Y = build_dataset(words)
    for i in range(max_steps):
        # minibatch construction
        ix = torch.randint(0, len(X), (batch_size,))
        in_batch = X[ix]
        out_batch = Y[ix]

        # Forward-Pass
        # to embedding vektor
        input_batch = C[in_batch]
        in_embcat = input_batch.view(input_batch.shape[0], -1)
        N1_out = torch.tanh(in_embcat @ W1 + b1)
        N2_out = N1_out @ W2 + b2
        loss = F.cross_entropy(N2_out, out_batch)
        if i % 100 == 0:
            print(f"Durchlauf: {i:5d} --- Loss:{loss.item():.4f}")

        # backprop
        for p in params:
            p.grad = None
        loss.backward()

        # update
        for p in params:
            assert p.grad is not None
            p.data += -learning_rate * p.grad


if __name__ == "__main__":
    main()
    
