import torch
import torch.nn as nn
import torch.nn.functional as F

import time

torch.random.manual_seed(42)

INT_TO_CHAR = [
    '\n', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '!', '\"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', ' '
];
VOCAB_SIZE = len(INT_TO_CHAR)

CORPUS_SIZE = 160000;
all_tokens = []
corpus = []

def read_dataset():
    global corpus
    global all_tokens
    with open("dataset.txt", 'r') as _f:
        for line in _f.readlines():
            corpus.extend(line)
    
    corpus = corpus[:CORPUS_SIZE]
    assert len(corpus) == CORPUS_SIZE
    for c in corpus:
        try:
            all_tokens.append(INT_TO_CHAR.index(c))
        except Exception as e:
            pass

read_dataset()


class NN(nn.Module):

    def __init__(self, vocab_size, embed_dim, input_dim, hidden_dim):
        super().__init__()
        self.embeding = nn.Embedding(vocab_size, embed_dim)
        self.W1 = nn.Linear(input_dim, hidden_dim, dtype=torch.float32)
        # self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.W2 = nn.Linear(hidden_dim, vocab_size, dtype=torch.float32)

    def forward(self, input_tokens):
        x = self.embeding(input_tokens).view(input_tokens.shape[0], -1)
        h = self.W1(x)
        # h = self.batch_norm(h)
        h_act = F.sigmoid(h)
        o = self.W2(h_act)
        return o
    
    def init_weights(self):
        torch.nn.init.xavier_normal_(self.embeding.weight, gain=2.0)
        torch.nn.init.xavier_normal_(self.W1.weight, gain=2.0)
        # torch.nn.init.zeros_(self.W1.bias)
        torch.nn.init.xavier_normal_(self.W2.weight, gain=2.0)
        # torch.nn.init.zeros_(self.W2.bias)


LR = 1e-1
B = 64
EMBED_DIM = 64
CONTEXT_LENGTH = 16
HIDDEN_DIM = 64
INPUT_DIM = CONTEXT_LENGTH * EMBED_DIM

model = NN(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM
)
# print("W1", model.W1.weight, model.W1.bias)
# print("W2", model.W2.weight, model.W2.bias)
# print("emb", model.embeding.weight)
# model.init_weights()
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.0)

MAX_STEPS = 300
start_time = time.time()
for i in range(MAX_STEPS):
    indices = torch.randint(0, len(all_tokens) - (CONTEXT_LENGTH + 1), (B, ))
    input_tokens, target_tokens = [], []
    for ind in indices:
        input_tokens.append(all_tokens[ind: ind + CONTEXT_LENGTH])
        target_tokens.append(all_tokens[ind + CONTEXT_LENGTH])


    input_tokens = torch.tensor(input_tokens)
    target_tokens = torch.tensor(target_tokens)
    # print(input_tokens, target_tokens)

    optimizer.zero_grad()
    o = model(input_tokens)
    loss = F.cross_entropy(o, target_tokens)
    print(i + 1, loss)
    loss.backward()
    optimizer.step()
    # break

print(f"Time taken: {time.time() - start_time} seconds")