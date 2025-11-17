import torch
import string

"""

This is to implement the counting method for bigrams. 

What do I need to do?


a. I need to first associate the characters with indexes. 
b. Once I do that, I need to generate the counts. I can generate the counts by doing what?
    I use zip(treatedName, treatedName[1:])
c. Then I need to 


"""



# let's first make a character association with indices and the other way around. 

charList = "a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z"

charArray = charList.split(",")
charToIntMapping = {}
intToCharMapping = {}

for index, char in enumerate(charArray):
    charToIntMapping[char] = index + 1
    intToCharMapping[index +1 ] = char
charToIntMapping["."] = 0
intToCharMapping[0] = "."


# ok now I want to go through the counts
names = []
with open('names.txt', 'r') as file:
    names = [line.strip() for line in file]


counts = torch.ones(27, 27)

for name in names:
    treatedName = "." + name + "."
    for ch1, ch2 in zip(treatedName, treatedName[1:]):
        ch1Index = charToIntMapping[ch1]
        ch2Index = charToIntMapping[ch2]
        counts[ch1Index, ch2Index ] += 1

# this now gives me the counts for the whole thing
# I want the probabilities

probs = counts / counts.sum(dim=1, keepdim=True)
row1 = probs[2]



# ok I now have the probabilities. So basically my model is "trained". I now want to generate actual names
# you want to start with the ., and given the . figure out which character comes next.

g = torch.Generator().manual_seed(50283810)

for i in range(20):
    output = ""
    ix = 0
    while True:
        probabilityDistributionOfNextCharGivenFirstChar = probs[ix]
        # now we want to sample the index from this vector
        nextCharIndex = torch.multinomial(probabilityDistributionOfNextCharGivenFirstChar, num_samples=1, generator=g).item()
        nextChar = intToCharMapping[nextCharIndex]
        if nextChar == ".":
            print(output)
            break
        output += nextChar
        ix = nextCharIndex




