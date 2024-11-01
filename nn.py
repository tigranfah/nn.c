import torch
import torch.nn as nn
import torch.nn.functional as F

torch.random.manual_seed(42)


vocab_size = 4;
INT_TO_CHAR = [
    '\n', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '!', '\"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', ' '
];
BOS_TOKEN = '$'
EOS_TOKEN = '@'
PAD_TOKEN = '#'
context_length = 4;
embed_dim = 2;
input_dim = context_length * embed_dim;
hidden_dim = 5
output_dim = vocab_size
B = 3
all_tokens = torch.randint(0, vocab_size, (B, context_length + 1))
input_tokens = all_tokens[:, :-1]
target_tokens = all_tokens[:, -1:].view(-1)
print(input_tokens, target_tokens)

embeddings = nn.Embedding(vocab_size, embed_dim)
# print(embeddings.weight)

x = embeddings(input_tokens).view(B, context_length * embed_dim)
x.retain_grad()
# x = torch.rand(B, input_dim, requires_grad=True)
y = F.one_hot(target_tokens, vocab_size)
# print(x, y_hot)
L1 = nn.Linear(input_dim, hidden_dim, bias=True)
L2 = nn.Linear(hidden_dim, output_dim, bias=True)
h = L1(x)
h_act = F.sigmoid(h)
o = L2(h_act)
h.retain_grad()
h_act.retain_grad()
o.retain_grad()
loss = F.cross_entropy(o, target_tokens)
loss.backward()
# print("o", o)
# print("o grad", o.grad)
# print("l1 grad", L1.weight.grad, L1.bias.grad)
# print("l2 grad", L2.weight.grad, L2.bias.grad)
# print("h act grad", h_act.grad)
# print("h grad", h.grad)
# print("x grad", x.grad)
print(embeddings.weight.grad)

# normal
W1 = L1.weight
b1 = L1.bias

W2 = L2.weight
b2 = L2.bias

# print(x)
h1 = x @ W1.T + b1
h1_act = F.sigmoid(h1)
# print(h1_relu)
o1 = h1_act @ W2.T + b2
q1 = F.softmax(o1, dim=-1)
# print("q1", q1)

o_grad = 1 / B * (q1 - y)
# print("o_grad", o_grad)
W2_grad = o_grad.T @ h1_act
b2_grad = o_grad.sum(dim=0)
# print(b2_grad)

def sigmoid_grad(x):
    return F.sigmoid(x) * (1 - F.sigmoid(x))

h1_act_grad = o_grad @ W2
# print("h1_act_grad", h1_act_grad)
h1_grad = h1_act_grad * sigmoid_grad(h1)
# print("h1_grad", h1_grad)
W1_grad = h1_grad.T @ x
b1_grad = h1_grad.sum(dim=0)

x_grad = h1_grad @ W1
# print("x_grad", x_grad)

# x
# {-0.3319f, -0.4785f, -0.2631f, -0.1786f, -1.2284f,  0.5294f, -0.3319f, -0.4785f},
# {-0.2631f, -0.1786f, -1.2284f,  0.5294f, -1.2284f,  0.5294f, -0.3319f, -0.4785f},
# {-0.3319f, -0.4785f, -0.3319f, -0.4785f, -0.3319f, -0.4785f, -0.3319f, -0.4785f}

# embed
# {-1.2284f,  0.5294f},
# { 1.2211f,  0.1511f},
# {-0.3319f, -0.4785f},
# {-0.2631f, -0.1786f}

# W2
# {-0.3008f,  0.1811f,  0.1601f,  0.3716f, -0.2310f},
# {-0.3049f,  0.2373f, -0.1808f,  0.2714f, -0.1061f},
# { 0.2558f, -0.3475f, -0.2257f,  0.1363f,  0.0945f},
# {-0.1140f,  0.2666f,  0.3040f, -0.3243f, -0.2388f}
# b2
# { 0.4095f, -0.1509f, -0.1585f, -0.4327f}

# W1
# {-0.3003f,  0.2730f,  0.0588f, -0.1148f,  0.2185f,  0.0551f,  0.2857f,  0.0387f},
# {-0.1115f,  0.0950f, -0.0959f,  0.1488f,  0.3157f,  0.2044f, -0.1546f,  0.2041f},
# { 0.0633f,  0.1795f, -0.2155f, -0.3500f, -0.1366f, -0.2712f,  0.2901f,  0.1018f},
# { 0.1464f,  0.1118f, -0.0062f,  0.2767f, -0.2512f,  0.0223f, -0.2413f,  0.1090f},
# {-0.1218f,  0.1083f, -0.0737f,  0.2932f, -0.2096f, -0.2109f, -0.2109f,  0.3180f}
# b1
# { 0.1178f,  0.3402f, -0.2918f, -0.3507f, -0.2766f}