import torch
import string
import torch.nn.functional as F

# I need to initialize a weights matrix, which is equivalent to the N that I have from the other script
char_to_index = {char: i+1 for i, char in enumerate(string.ascii_lowercase)}
index_to_char = {i+1 : char for i, char in enumerate(string.ascii_lowercase)}
char_to_index["."] = 0
index_to_char[0] = "."

W = torch.randn(54,27, requires_grad=True)

# now I need to represent the training examples 
with open('names.txt', 'r') as file:
    names = [line.strip() for line in file]

batch = torch.zeros(3, 27)
xs = []
ys = []
x = []

for name in names[:50]:
    treatedName = "." + name + "."
    for ch1, ch2, ch3 in zip(treatedName, treatedName[1:], treatedName[2:]):
        ix1 = char_to_index[ch1]
        ix2 = char_to_index[ch2]
        ix3 = char_to_index[ch3]
        one_hot_ch1 = torch.zeros(27)
        one_hot_ch2 = torch.zeros(27)
        one_hot_ch3 = torch.zeros(27)
        one_hot_ch1[ix1] = 1
        one_hot_ch2[ix2] = 1
        one_hot_ch3[ix3] = 1
        x = torch.cat([one_hot_ch1, one_hot_ch2])
        xs.append(x)
        ys.append(one_hot_ch3)


X = torch.stack(xs)  # Convert list of tensors to tensor: shape (m, 54)
Y = torch.stack(ys)  # Convert list of tensors to tensor: shape (m, 27)
nTrainingExamples = len(ys)
# ok so i now have my training set. 



for k in range(100):
    logits = X @ W  # Raw scores (no exp yet)
    probs = logits.softmax(dim=1)  # Apply softmax: exp / sum(exp)
    # calculate the loss
    target_indices = Y.argmax(dim=1)
    loss = -probs[torch.arange(nTrainingExamples),target_indices].log().mean()
    print(loss.item())
    # now I have the loss I need to update the weights

    W.grad = None
    loss.backward()

    W.data += -1 *W.grad

probs = X @ W
probs = logits.softmax(dim = 1)


for i in range(10):
    out = []
    ix =0
    ix1 = 0
    while True:
        # p = N[ix, ix1].float()
        # p = p / p.sum()
        p = probs[ix,ix1].float()
        # I have the probability of this 
        ix2 = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(index_to_char[ix2])
        if ix2 == 0:
            break
        ix = ix1
        ix1 = ix2

    print(''.join(out))


X = torch.stack(xs)
Y = torch.stack(ys)

# now I want to use the model







