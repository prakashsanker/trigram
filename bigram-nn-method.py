from math import log
import torch
import string

"""
1. Generate a random set of weights here for W
2. Figure out how to represent the training examples as a set of one hot vectors
3. Do the forward pass
4. Calculate the gradient via .backward()
5. Figure out the loss
6. Update the weights

"""


charList = "a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z"

charArray = charList.split(",")
charToIntMapping = {}
intToCharMapping = {}

for index, char in enumerate(charArray):
    charToIntMapping[char] = index + 1
    intToCharMapping[index +1 ] = char
charToIntMapping["."] = 0
intToCharMapping[0] = "."


names = []
with open('names.txt', 'r') as file:
    names = [line.strip() for line in file]


W = torch.rand(27,27, requires_grad=True)
xs = []
ys = []
nExamples = 0
for name in names:
    treatedName = "." + name + "."
    for ch1, ch2 in zip(treatedName, treatedName[1:]):
        ch1Index = charToIntMapping[ch1]
        ch2Index = charToIntMapping[ch2]
        x_one_hot = torch.zeros(27)
        x_one_hot[ch1Index] = 1
        y_one_hot = torch.zeros(27)
        y_one_hot[ch2Index] = 1
        xs.append(x_one_hot)
        ys.append(y_one_hot)
        nExamples += 1
X = torch.stack(xs)
Y = torch.stack(ys)

# ok I have the weights initially, the training examples, and the predictions


# I need to do a forward pass, in order to do the calculation
# Figure out the loss (log likelihood)
# Then update the weights
# This needs to repeat for a number of steps




for i in range(1000):
    # So we go through this process 10 times
    logits = X @ W # this is similar to our counts matrix in the counting method, but it gets us the counts for that particular initial character
    probs = logits.softmax(dim=1) # we convert to probabiities from raw counts. 
    target_indices = Y.argmax(dim=1) # these are the characters that are following
    # I want to figure out the loss for each example 
    # I know that X is also a set of one hot vectors
    # for each X, I want to figure out the probability of the next
    loss = -probs[torch.arange(nExamples),target_indices].log().mean() # so here we go, for each example, we've already multiplied to figure out the counts/probs for that particular character. What is the most likely next character? Take the logg of this, then average that, then negate. This is a score for the prediction. This is what you want to minimize in gradient descent. 
    print(loss.item())
    W.grad = None
    loss.backward()
    W.data += -1 * W.grad


g = torch.Generator().manual_seed(50283810)



for i in range(10):
    output = ""
    x = torch.zeros(27)
    nextIndex = 0
    while True:
        x[nextIndex] = 1
        logits = x @ W
        probs = logits.softmax(dim=0)
        nextCharIndex = torch.multinomial(probs, num_samples=1, generator=g).item()
        if nextCharIndex == 0:
            print(output)
            break
        charToAdd = intToCharMapping[nextCharIndex]
        output += charToAdd
        x = torch.zeros(27)
        nextIndex = nextCharIndex








