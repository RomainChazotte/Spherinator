import torch
mask = torch.zeros(3)
mask[0] +=1
mask[2] +=1
mask = mask.bool()
x = torch.zeros(3,3,2)
y = torch.ones(2,2)
x[:,mask] = y
print(x)