import torch

tens = torch.zeros(120)
m = torch.nn.BatchNorm2d(5, affine=False)
for i in range(120):
    tens[i] = i
tens = tens.view(3,5,2,4)
print(tens)
print(m(tens))
