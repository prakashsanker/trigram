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


W = torch.rand(54,27, requires_grad=True)
xs = []
ys = []
nNames = len(names)
developmentSetIndex = 2500
trainingSetIndex = 29533

development_xs = []
testing_xs = []
development_ys = []
testing_ys = []
nDevelopmentExamples = 0
nTestingExamples = 0



nExamples = 0
for name in names[0:developmentSetIndex]:
    treatedName = "." + name + "."
    for ch1, ch2, ch3 in zip(treatedName, treatedName[1:], treatedName[2:]):
        ch1Index = charToIntMapping[ch1]
        ch2Index = charToIntMapping[ch2]
        ch3Index = charToIntMapping[ch3]
        x_one_hot = torch.zeros(54)
        x_one_hot[ch1Index] = 1
        x_one_hot[ch2Index + 27] = 1
        y_one_hot = torch.zeros(27)
        y_one_hot[ch3Index] = 1
        xs.append(x_one_hot)
        ys.append(y_one_hot)
        nExamples += 1
X = torch.stack(xs)
Y = torch.stack(ys)


for name in names[developmentSetIndex:trainingSetIndex]:
    treatedName = "." + name + "."
    for ch1, ch2, ch3 in zip(treatedName, treatedName[1:], treatedName[2:]):
        ch1Index = charToIntMapping[ch1]
        ch2Index = charToIntMapping[ch2]
        ch3Index = charToIntMapping[ch3]
        x_one_hot = torch.zeros(54)
        x_one_hot[ch1Index] = 1
        x_one_hot[ch2Index + 27] = 1
        y_one_hot = torch.zeros(27)
        y_one_hot[ch3Index] = 1
        development_xs.append(x_one_hot)
        development_ys.append(y_one_hot)
        nDevelopmentExamples += 1

DevelopmentX = torch.stack(development_xs)
DevelopmentY = torch.stack(development_ys)


for name in names[trainingSetIndex:]:
    treatedName = "." + name + "."
    for ch1, ch2, ch3 in zip(treatedName, treatedName[1:], treatedName[2:]):
        ch1Index = charToIntMapping[ch1]
        ch2Index = charToIntMapping[ch2]
        ch3Index = charToIntMapping[ch3]
        x_one_hot = torch.zeros(54)
        x_one_hot[ch1Index] = 1
        x_one_hot[ch2Index + 27] = 1
        y_one_hot = torch.zeros(27)
        y_one_hot[ch3Index] = 1
        testing_xs.append(x_one_hot)
        testing_ys.append(y_one_hot)
        nTestingExamples += 1
TestingX = torch.stack(testing_xs)
TestingY = torch.stack(testing_ys)






# ok I have the weights initially, the training examples, and the predictions


# I need to do a forward pass, in order to do the calculation
# Figure out the loss (log likelihood)
# Then update the weights
# This needs to repeat for a number of steps


for i in range(500):
    # So we go through this process 10 times
    logits = X @ W # this is similar to our counts matrix in the counting method, but it gets us the counts for that particular initial character
    # # find the index
    # # total_loss = 0
    # # for j in range(nExamples):
    # #     row_vector_indices = W[j].argmax(dim=1)
    # #     # now I want to 
    # #     vector_1_distribution = W[j][row_vector_indices[0]].softmax()
    # #     vector_2_distribution = W[j][row_vector_indices[1]].softmax()
    # #     total_loss += -vector_1_distribution
    # probs = logits.softmax(dim=1) # we convert to probabiities from raw counts. 
    target_indices = Y.argmax(dim=1) # these are the characters that are following
    # I want to figure out the loss for each example 
    # I know that X is also a set of one hot vectors
    # for each X, I want to figure out the probability of the next
    # loss = -probs[torch.arange(nExamples),target_indices].log().mean()  + 0.3*(W**2).mean()# so here we go, for each example, we've already multiplied to figure out the counts/probs for that particular character. What is the most likely next character? Take the logg of this, then average that, then negate. This is a score for the prediction. This is what you want to minimize in gradient descent. 
    loss = torch.nn.functional.cross_entropy(logits, target_indices)
    print(loss.item())
    W.grad = None
    loss.backward()
    W.data += -50 * W.grad

g = torch.Generator().manual_seed(50283810)

# Evaluate against the development set

logits = DevelopmentX @ W
probs = logits.softmax(dim=1)
target_indices = DevelopmentY.argmax(dim=1)
loss = -probs[torch.arange(nDevelopmentExamples),target_indices].log().mean()
print("DEVELOPMENT SET LOSS")
print(loss.item())

logits = TestingX @ W
probs = logits.softmax(dim=1)
target_indices = TestingY.argmax(dim=1)
loss = -probs[torch.arange(nTestingExamples),target_indices].log().mean()
print("Testing SET LOSS")
print(loss.item())



# for i in range(10):
#     output = ""
#     x = torch.zeros(54)
#     char1Index = 0
#     char2Index = 0
#     while True:
#         x[char1Index] = 1
#         x[char2Index + 27] = 1
#         logits = x @ W
#         probs = logits.softmax(dim=0)
#         nextCharIndex = torch.multinomial(probs, num_samples=1, generator=g).item()
#         if nextCharIndex == 0:
#             print(output)
#             break
#         charToAdd = intToCharMapping[nextCharIndex]
#         output += charToAdd
#         x = torch.zeros(54)
#         char1Index = char2Index
#         char2Index = nextCharIndex








