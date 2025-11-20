import torch
import string


# First let's set up the training data


charList = "a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z"

charArray = charList.split(",")
charToIntMapping = {}
intToCharMapping = {}

for index, char in enumerate(charArray):
    charToIntMapping[char] = index + 1
    intToCharMapping[index +1 ] = char
charToIntMapping["."] = 0
intToCharMapping[0] = "."


with open('names.txt', 'r') as file:
    names = [line.strip() for line in file]

# we have to split the names array into a training set, a dev set and a test set.
generator = torch.Generator().manual_seed(50283810)
xs = []
ys = []
x_dev = []
y_dev = []
x_test = []
y_test= []

embedding_vector_dim = 2
context_size = 3
number_of_neurons = 300
step_size = 0.01
n_steps =2000

total_training_data_size = len(names)
training_set_index = int(0.8*total_training_data_size)
dev_set_index = int(0.9*total_training_data_size)

def create_data_set(startIndex, endIndex, names):
    xs = []
    ys = []
    print("CREATE_DATA SET")
    print(startIndex)
    print(endIndex)
    names_subset = names[startIndex: endIndex]
    for name in names_subset:
        treatedName = "." + name + "."
        context= [".", ".", "."]
        for current_char, next_char in zip(treatedName, treatedName[1:]):
            xs.append(context.copy())
            ys.append(next_char)
            context = context[1:] + [current_char]
    xs_int = [[charToIntMapping[char] for char in context] for context in xs]
    ys_int = [charToIntMapping[char] for char in ys]
    return xs_int, ys_int


xs_int, ys_int = create_data_set(0, training_set_index, names)
x_dev, y_dev = create_data_set(training_set_index +1, dev_set_index, names)
x_test, y_test = create_data_set(dev_set_index +1, len(names) -1, names)

# now I want to initialize the embedding matrix

C = torch.rand(27,embedding_vector_dim, requires_grad=True, generator=generator)



# input into the neural net is 
X = torch.tensor(xs_int)
Y = torch.tensor(ys_int)
XTest = torch.tensor(x_test)
YTest = torch.tensor(y_test)

print("X SHAPE")
print(X.shape)
print("Y SHAPE")
print(Y.shape)


# now I want to initialize the weights and biases

W1 = torch.rand(context_size*embedding_vector_dim, number_of_neurons, requires_grad=True, generator=generator)
b1 = torch.rand(1, number_of_neurons, requires_grad=True, generator=generator)
W2 = torch.rand(number_of_neurons,27, requires_grad=True, generator=generator)
b2 = torch.rand(1, 27, requires_grad=True, generator=generator)
parameters = [C, W1, W2, b1, b2]

input_to_llm = C[X]
# print("C[X] shape")
# print(input_to_llm.shape)
# print("TRAINING SET LENGTH")
# print(len(xs_int))



# now I want to actually do the gradient descent and update the gradients

for i in range(n_steps):
    # layer 1 of the neural net
    input_to_llm = C[X]
    # print("X SHAPE")
    # print(X.shape[0])
    treated_input_to_llm = input_to_llm.view(X.shape[0], context_size*embedding_vector_dim)
    h1 = treated_input_to_llm @ W1 + b1
    # print("SHAPES")
    # print(treated_input_to_llm.shape)
    # print(W1.shape)
    # print(b1.shape)
        # m by 10 * 10 by 100 + 1 by 100
        # h2 should be m by 100. So in this case, m should be the number of training examples, it should be unaffected by the embedding vector?
    h2 = h1 @ W2 + b2

    loss = torch.nn.functional.cross_entropy(h2, Y)
    # now we want to update the gradients
    print(loss.item())
    for param in parameters:
        param.grad = None
    loss.backward()
    for param in parameters:
        param.data += -step_size* param.grad

# # ok so here I have the parameters of the net optimized
# # I want to now calculate the loss for the test set? 


def calculate_test_set_loss():
    input_to_llm = C[XTest]
    treated_input_to_llm = input_to_llm.view(-1, context_size*embedding_vector_dim)
    h1 = treated_input_to_llm @ W1 + b1
    interim = h1 @ W2
    h2 = h1 @ W2 + b2
    loss = torch.nn.functional.cross_entropy(h2, YTest)
    # now we want to update the gradients
    print("Test SET LOSS")
    print(loss.item())
calculate_test_set_loss()









    

    















