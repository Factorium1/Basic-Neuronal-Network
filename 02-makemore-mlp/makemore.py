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
        for ch in w + ".":
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
W1 = torch.randn((embd_dim * block_size, hidden_dim)) / (embd_dim * block_size) ** 0.5
b1 = torch.zeros(hidden_dim)
W2 = torch.randn((hidden_dim, alphabet_size)) * 0.01
b2 = torch.zeros(alphabet_size)

params = [C, W1, W2, b1, b2]
for p in params:
    p.requires_grad = True

# work
max_steps = 20_000
batch_size = 32
learning_rate = 0.1


def forward_pass(in_batch):
    # Forward-Pass
    # to embedding vektor
    input_batch = C[in_batch]
    in_embcat = input_batch.view(
        input_batch.shape[0], -1
    )  # shape from e.g.[32,3,10] -> [32, 30]
    N1_out = torch.tanh(in_embcat @ W1 + b1)
    N2_out = N1_out @ W2 + b2
    return N2_out


def main():
    X, Y = build_dataset(words)
    for i in range(max_steps):
        # minibatch construction
        ix = torch.randint(0, len(X), (batch_size,))
        in_batch = X[ix]
        out_batch = Y[ix]

        logits = forward_pass(in_batch)
        loss = F.cross_entropy(logits, out_batch)

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

    out = ["" for _ in range(10)]
    with torch.no_grad():
        for i in range(10):
            context = [0] * block_size
            while True:
                logits = forward_pass(torch.tensor([context]))
                wsh = F.softmax(logits, -1)
                next_index = int(torch.multinomial(wsh, 1).item())
                if next_index == 0:
                    break
                out[i] += itos[next_index]
                context = context[1:] + [next_index]

    for o in out:
        print(o)
