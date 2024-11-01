import torch
import torch.nn as nn
import torch.nn.functional as F

import time
from pprint import pprint

torch.set_printoptions(precision=6, sci_mode=False)

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
        self.x = self.embeding(input_tokens).view(input_tokens.shape[0], -1)
        self.h = self.W1(self.x)
        # h = self.batch_norm(h)
        self.h_act = F.sigmoid(self.h)
        # print(h_act)
        # print(self.W2.weight)
        self.o = self.W2(self.h_act)
        # print(o)
        self.x.retain_grad()
        self.h.retain_grad()
        self.h_act.retain_grad()
        self.o.retain_grad()
        return self.o
    
    def _read_weights(self, _f, layer):
        line = _f.readline().rstrip("\n")
        print("reading weight", line)
        weights_list = []
        for _ in range(layer.weight.shape[0]):
            line = _f.readline().rstrip(" \n")
            weights_list.append([float(v) for v in line.split(" ")])
        
        return torch.tensor(weights_list)
    
    def init_weights(self):
        # xavier init
        # self.embedding.weight = torch.normal(0, torch.sqrt(2 / sum(self.embeding.weight.shape)))
        # self.W1.weight = torch.normal(0, torch.sqrt(2 / sum(self.W1.weight.shape)))
        # self.W2.weight = torch.normal(0, torch.sqrt(2 / sum(self.W2.weight.shape)))
        with torch.no_grad():
            with open("C_weight.txt", "r") as _f:
                self.embeding.weight = nn.Parameter(self._read_weights(_f, self.embeding))
                self.W1.weight = nn.Parameter(self._read_weights(_f, self.W1))
                self.W1.bias = nn.Parameter(torch.zeros_like(self.W1.bias))
                self.W2.weight = nn.Parameter(self._read_weights(_f, self.W2))
                self.W2.bias = nn.Parameter(torch.zeros_like(self.W2.bias))

        # print("embeddings")
        # print(self.embeding.weight)
        # print("W1")
        # print(self.W1.weight)
        # print("W2")
        # print(self.W2.weight)


LR = 1e-1
MAX_STEPS = 1000
B = 128
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
model.init_weights()
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.0)
log_file = open("pytorch.log", "w")

start_time = time.time()
for step in range(1, MAX_STEPS + 1):
    # indices = torch.randint(0, len(all_tokens) - (CONTEXT_LENGTH + 1), (B, ))
    indices = torch.tensor([0, 25, 50, 75, 100])
    input_tokens, target_tokens = [], []
    for ind in indices:
        input_tokens.append(all_tokens[ind: ind + CONTEXT_LENGTH])
        target_tokens.append(all_tokens[ind + CONTEXT_LENGTH])

    # pprint(input_tokens)
    # pprint(target_tokens)
    input_tokens = torch.tensor(input_tokens)
    target_tokens = torch.tensor(target_tokens)
    # print(input_tokens, target_tokens)

    optimizer.zero_grad()
    o = model(input_tokens)
    loss = F.cross_entropy(o, target_tokens)
    line = f"step {step}, loss {loss.item():.6f}"
    print(line)
    log_file.write(line + "\n")
    loss.backward()
    optimizer.step()
    # print(model.W2.weight.grad)
    if step == 2:
        break

print(f"Time taken: {time.time() - start_time} seconds")